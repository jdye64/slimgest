import argparse
import time
import torch
import numpy as np
import pypdfium2 as pdfium
import multiprocessing as mp
from queue import SimpleQueue
from statistics import mean, stdev
import random

# ------------------------------------------------------------
# GPU cropping helper
# ------------------------------------------------------------

def crop_gpu(image_tensor, bbox):
    """
    image_tensor: CHW on CUDA
    bbox: (x1, y1, x2, y2)
    """
    x1, y1, x2, y2 = bbox
    return image_tensor[:, y1:y2, x1:x2]


# ------------------------------------------------------------
# Worker for PDFium page rasterization
# ------------------------------------------------------------
def worker_raster(pdf_path, page_indices, out_queue):
    pdf = pdfium.PdfDocument(pdf_path)
    for idx in page_indices:
        page = pdf.get_page(idx)
        bitmap = page.render(scale=1.0, rotation=0, color=True, grayscale=False)
        arr = bitmap.to_numpy()  # H x W x 4 RGBA
        arr = arr[...,:3]        # drop alpha
        out_queue.put((idx, arr))
        page.close()
    out_queue.put(None)


# ------------------------------------------------------------
# Benchmark runner
# ------------------------------------------------------------

def benchmark(pdf_path, num_workers=4, max_pages=None, device="cuda"):
    pdf = pdfium.PdfDocument(pdf_path)
    total_pages = len(pdf)
    if max_pages:
        total_pages = min(total_pages, max_pages)

    print(f"[INFO] Benchmarking {total_pages} pages with {num_workers} workers…")

    # Assign pages to workers
    pages_per_worker = (total_pages + num_workers - 1) // num_workers
    worker_assignments = [
        list(range(i * pages_per_worker, min((i+1) * pages_per_worker, total_pages)))
        for i in range(num_workers)
    ]

    # Spawn queue and processes
    q = SimpleQueue()
    procs = [
        mp.Process(target=worker_raster, args=(pdf_path, assignment, q))
        for assignment in worker_assignments
    ]

    for p in procs:
        p.start()

    # Storage
    raster_times = []
    h2d_times = []
    crop_times = []
    page_latencies = []
    processed = 0

    torch.cuda.synchronize()

    t0_global = time.time()

    # Fake detection bboxes
    def random_bbox(h, w):
        x1 = random.randint(0, w // 3)
        y1 = random.randint(0, h // 3)
        x2 = random.randint(w // 2, w - 1)
        y2 = random.randint(h // 2, h - 1)
        return (x1, y1, x2, y2)

    finished_workers = 0
    while finished_workers < num_workers:
        item = q.get()
        if item is None:
            finished_workers += 1
            continue

        idx, arr = item
        processed += 1

        # -------------------------
        # Rasterization
        # -------------------------
        t_r0 = time.time()
        # Already rasterized in worker
        t_r1 = time.time()
        raster_times.append(t_r1 - t_r0)

        # -------------------------
        # Upload to GPU (async)
        # -------------------------
        t_h2d0 = time.time()
        t_cpu = torch.from_numpy(arr).permute(2, 0, 1).contiguous()
        t_gpu = t_cpu.to(device, non_blocking=True)
        torch.cuda.synchronize()
        t_h2d1 = time.time()
        h2d_times.append(t_h2d1 - t_h2d0)

        # -------------------------
        # GPU crop simulate detection
        # -------------------------
        h, w = arr.shape[0], arr.shape[1]
        bbox = random_bbox(h, w)

        t_c0 = time.time()
        crop_gpu(t_gpu, bbox)
        torch.cuda.synchronize()
        t_c1 = time.time()
        crop_times.append(t_c1 - t_c0)

        page_latencies.append((t_c1 - t_r0))

    t1_global = time.time()

    for p in procs:
        p.join()

    # ------------------------------------------------------------
    # Report
    # ------------------------------------------------------------
    print("\n========== PERFORMANCE REPORT ==========")
    print(f"Total pages processed:    {processed}")
    print(f"Total time:               {t1_global - t0_global:.4f} sec")
    print(f"Throughput:               {processed / (t1_global - t0_global):.2f} pages/sec")

    print("\n--- Breakdown (avg ms) ---")
    print(f"Rasterization (CPU):      {mean(raster_times)*1000:.3f} ms ± {stdev(raster_times)*1000 if len(raster_times)>1 else 0:.3f}")
    print(f"H2D Upload (GPU):         {mean(h2d_times)*1000:.3f} ms ± {stdev(h2d_times)*1000 if len(h2d_times)>1 else 0:.3f}")
    print(f"GPU Cropping:             {mean(crop_times)*1000:.3f} ms ± {stdev(crop_times)*1000 if len(crop_times)>1 else 0:.3f}")
    print(f"End-to-end per page:      {mean(page_latencies)*1000:.3f} ms")

    print("\n--- Estimated Limits ---")
    print(f"Max pages/sec from upload: {1.0 / mean(h2d_times):.1f}")
    print(f"Max pages/sec from raster: {1.0 / mean(raster_times):.1f}")
    print(f"Max pages/sec from total:  {1.0 / mean(page_latencies):.1f}")

    print("========================================\n")

    # Optimization suggestions
    print("RECOMMENDATIONS:")
    if mean(h2d_times) > 0.003:
        print("• Enable pinned CPU memory for faster H2D transfers")
    if mean(raster_times) > 0.005:
        print("• Increase parallel workers or use multiple PDFs in flight")
    if mean(crop_times) > 0.001:
        print("• Fuse cropping with model preprocessing on GPU")



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pdf", required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--max-pages", type=int, default=None)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    benchmark(args.pdf, args.workers, args.max_pages, args.device)
