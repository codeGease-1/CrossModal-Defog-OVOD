"""Synchronized inference timing with explicitly bounded preprocessing scope."""
import io
from pathlib import Path
import time
import numpy as np
from PIL import Image
import torch
from .baselines import build_method, resolve_device
from .common import fresh_directory, write_json, environment, NotAvailable
from .report import csv_table


def latency_summary(seconds, batch_size=1):
    values = np.asarray(seconds, dtype=np.float64)
    if values.size == 0 or not np.isfinite(values).all() or (values <= 0).any() or batch_size < 1:
        raise ValueError('Timing samples must be finite and positive')
    return {'iterations': len(values), 'mean_ms': float(values.mean() * 1000),
            'p50_ms': float(np.quantile(values, .5) * 1000),
            'p95_ms': float(np.quantile(values, .95) * 1000),
            'images_per_second': float(batch_size * len(values) / values.sum()),
            'raw_latency_ms': (values * 1000).tolist()}


def measure(operation, device, warmup, repeats, batch_size=1):
    if warmup < 1 or repeats < 2:
        raise ValueError('At least one warmup and two measured iterations required')
    def synchronize():
        if device.type == 'cuda':
            torch.cuda.synchronize(device)
    with torch.inference_mode():
        for _ in range(warmup):
            result = operation()
            synchronize()
            del result
        if device.type == 'cuda':
            baseline = torch.cuda.memory_allocated(device)
            torch.cuda.reset_peak_memory_stats(device)
        samples = []
        for _ in range(repeats):
            synchronize()
            start = time.perf_counter()
            result = operation()
            synchronize()
            samples.append(time.perf_counter() - start)
            del result
        memory = {'status': 'N/A', 'reason': 'CUDA allocator memory does not apply to CPU',
                  'peak_allocated_bytes': None, 'peak_reserved_bytes': None, 'incremental_allocated_bytes': None}
        if device.type == 'cuda':
            peak = torch.cuda.max_memory_allocated(device)
            memory = {'status': 'PASS', 'peak_allocated_bytes': peak,
                      'peak_reserved_bytes': torch.cuda.max_memory_reserved(device),
                      'incremental_allocated_bytes': peak - baseline,
                      'scope': 'PyTorch allocator; includes model/input, excludes driver/other processes'}
    return {**latency_summary(samples, batch_size), 'memory': memory}


def convolution_linear_macs(model, tensor):
    """Partial operation count. Excludes normalization, pooling, activations, etc."""
    total = 0
    handles = []
    def hook(layer, inputs, output):
        nonlocal total
        if isinstance(layer, (torch.nn.Conv1d, torch.nn.Conv2d, torch.nn.Conv3d)):
            total += output.numel() * (layer.in_channels // layer.groups) * int(np.prod(layer.kernel_size))
        elif isinstance(layer, torch.nn.Linear):
            total += output.numel() * layer.in_features
    for layer in model.modules():
        if isinstance(layer, (torch.nn.Conv1d, torch.nn.Conv2d, torch.nn.Conv3d, torch.nn.Linear)):
            handles.append(layer.register_forward_hook(hook))
    try:
        with torch.inference_mode():
            model(tensor)
    finally:
        for handle in handles:
            handle.remove()
    return {'status': 'PASS' if handles else 'N/A', 'macs': int(total) if handles else None,
            'approx_flops_2x_macs': int(total * 2) if handles else None,
            'coverage': 'Conv1d/2d/3d and Linear only; excludes bias, norm, pooling, resize, activations and elementwise ops; NOT total FLOPs'}


def benchmark_performance(output, config, devices=None):
    output = fresh_directory(output)
    options = config['performance']
    batch_size = options.get('batch_size', 1)
    if batch_size != 1:
        raise ValueError('This baseline latency protocol currently requires batch_size=1')
    rows = []
    for device_name in devices or options['devices']:
        for size in options['sizes']:
            for name in config['methods']:
                record = {'method': name, 'device': device_name, 'input_size': size, 'batch_size': batch_size,
                          'warmup': options['warmup'], 'repeats': options['repeats'], 'precision': 'float32',
                          'threads': config.get('threads', 1)}
                try:
                    device = resolve_device(device_name)
                    # Reserved memory from the previous method must not inflate
                    # this method's allocator peak. Includes prior validation.
                    if device.type == 'cuda':
                        torch.cuda.empty_cache()
                    try:
                        model, info = build_method(name, device, config)
                    except NotAvailable:
                        if name != 'student' or not options.get('architecture_only_when_untrained'):
                            raise
                        model, info = build_method(name, device, config, architecture_only=True)
                    record.update(info)
                    # Fixed synthetic input is for engineering cost only, never accuracy.
                    pixels = np.random.default_rng(config['seed']).integers(0, 256, (size, size, 3), dtype=np.uint8)
                    tensor = torch.from_numpy(pixels.transpose(2, 0, 1).copy()).float().unsqueeze(0).div(255).to(device)
                    encoded = io.BytesIO()
                    Image.fromarray(pixels).save(encoded, format='PNG')
                    png = encoded.getvalue()
                    def forward():
                        return model(tensor)
                    def end_to_end():
                        with Image.open(io.BytesIO(png)) as image:
                            image = image.convert('RGB').resize((size, size), Image.Resampling.BILINEAR)
                            array = np.asarray(image).copy()
                        x = torch.from_numpy(array.transpose(2, 0, 1).copy()).float().unsqueeze(0).div(255).to(device)
                        return model(x).cpu().numpy()
                    # Verify shape/finite values once, outside timing.
                    with torch.inference_mode():
                        sample = forward()
                        if tuple(sample.shape) != (1, 1, size, size) or not torch.isfinite(sample).all():
                            raise ValueError('Performance target failed shape/finite validation')
                    del sample
                    for phase, operation in (('forward', forward), ('end_to_end', end_to_end)):
                        result = measure(operation, device, options['warmup'], options['repeats'], batch_size)
                        rows.append({**record, 'status': 'PASS', 'phase': phase, **result,
                                     'input_kind': 'seeded random RGB; engineering cost only',
                                     'scope': 'device-resident FP32 tensor -> prediction' if phase == 'forward'
                                     else 'in-memory PNG decode + RGB resize + FP32/255 + device transfer + forward + CPU return; excludes disk/network I/O'})
                    macs = convolution_linear_macs(model, tensor)
                    for row in rows[-2:]:
                        row['operation_count'] = macs
                    print(f'PASS performance: {record["method"]} {device_name} {size}', flush=True)
                    del model, tensor
                    if device.type == 'cuda':
                        torch.cuda.empty_cache()
                except NotAvailable as error:
                    rows.append({**record, 'status': 'NOT RUN', 'reason': str(error), 'phase': 'all'})
                except (ValueError, RuntimeError) as error:
                    rows.append({**record, 'status': 'FAIL', 'reason': str(error), 'phase': 'all'})
    status = 'FAIL' if any(r['status'] == 'FAIL' for r in rows) else 'PASS' if any(r['status'] == 'PASS' for r in rows) else 'NOT RUN'
    result = {'status': status, 'environment': environment(config['seed'], config), 'measurements': rows,
              'warning': 'Device-specific descriptive measurements. Random-weight student is architecture-only, not an accuracy baseline.'}
    write_json(output / 'performance.json', result)
    csv_table(output / 'performance.csv', [{k: v for k, v in row.items() if k not in ('raw_latency_ms', 'checkpoint_config', 'parameters')}
                                          for row in rows])
    lines = ['# 雾密度模块工程性能', '', f'执行状态：{status}。计时精度 FP32，batch=1；预热和重复次数见 JSON。',
             '', '端到端包含内存 PNG 解码、RGB 转换/缩放、张量转换、传输、前向和回传，不含磁盘、网络、模型初始化。吞吐是该串行 batch=1 协议下的实测吞吐。',
             '', '| 方法 | 设备 | 尺寸 | 范围 | 状态 | P50 ms | P95 ms | images/s | 参数量 |',
             '|---|---|---:|---|---|---:|---:|---:|---:|']
    for row in rows:
        def number(k):
            return f'{row[k]:.3f}' if row.get(k) is not None else 'NOT RUN'
        lines.append(f'| {row["method"]} | {row["device"]} | {row["input_size"]} | {row["phase"]} | {row["status"]} | {number("p50_ms")} | {number("p95_ms")} | {number("images_per_second")} | {row.get("learned_parameters", "NOT RUN")} |')
    lines += ['', '显存与原始逐次延迟见 `performance.json`。CPU 显存为 N/A；CUDA 缺失为 NOT RUN。MACs 仅统计 Conv/Linear，不能冒充完整 FLOPs。',
              '', '`student_architecture_only` 使用未训练的相同网络，仅反映结构计算成本，其正确性仍是 NOT EVALUATED。设备不同的数字不能直接证明算法优劣。']
    (output / 'report.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    return result
