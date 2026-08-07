// ── Global State Variables ──────────────────────────────────────────
let selectedFiles = [];
let dualLogsMap = {};
let lastMetrics = null;

// ── DOM Initialization ─────────────────────────────────────────────
document.addEventListener('DOMContentLoaded', () => {
    const uploadZone = document.getElementById('upload-zone');
    const fileInput = document.getElementById('image-file-input') || document.getElementById('file-input');
    const singleFileInput = document.getElementById('single-file-input');
    const processBtn = document.getElementById('process-btn');
    const uploadIdleState = document.getElementById('upload-idle-state');
    const uploadStatusCard = document.getElementById('upload-status-card');
    const uploadErrorBanner = document.getElementById('upload-error-banner');

    if (uploadZone) {
        uploadZone.addEventListener('dragover', (e) => {
            e.preventDefault();
            uploadZone.classList.add('border-amber-500/80', 'bg-amber-500/5');
        });
        uploadZone.addEventListener('dragleave', () => {
            uploadZone.classList.remove('border-amber-500/80', 'bg-amber-500/5');
        });
        uploadZone.addEventListener('drop', async (e) => {
            e.preventDefault();
            uploadZone.classList.remove('border-amber-500/80', 'bg-amber-500/5');
            const files = await getAllFilesFromDataTransfer(e.dataTransfer);
            handleFileSelect(files);
        });
    }

    if (fileInput) fileInput.addEventListener('change', (e) => handleFileSelect(Array.from(e.target.files)));
    if (singleFileInput) singleFileInput.addEventListener('change', (e) => handleFileSelect(Array.from(e.target.files)));

    if (processBtn) processBtn.addEventListener('click', processImages);

    window.addEventListener('keydown', (e) => {
        if (e.key === 'Escape') closeDetailModal();
    });

    loadGroundTruth();
});

// ── File Selection & Traversal ─────────────────────────────────────
async function getAllFilesFromDataTransfer(dataTransfer) {
    const files = [];
    const items = dataTransfer.items;

    async function traverse(item, path = '') {
        return new Promise((resolve) => {
            if (item.isFile) {
                item.file((file) => {
                    Object.defineProperty(file, 'webkitRelativePath', { value: path + file.name, writable: false });
                    files.push(file);
                    resolve();
                });
            } else if (item.isDirectory) {
                const reader = item.createReader();
                reader.readEntries(async (entries) => {
                    for (let entry of entries) await traverse(entry, path + item.name + '/');
                    resolve();
                });
            } else resolve();
        });
    }

    if (items && items.length > 0) {
        const promises = [];
        for (let i = 0; i < items.length; i++) {
            const entry = items[i].webkitGetAsEntry ? items[i].webkitGetAsEntry() : null;
            if (entry) promises.push(traverse(entry));
        }
        await Promise.all(promises);
    }
    return files.length > 0 ? files : Array.from(dataTransfer.files || []);
}

function handleFileSelect(files) {
    const uploadIdleState = document.getElementById('upload-idle-state');
    const uploadStatusCard = document.getElementById('upload-status-card');
    const uploadErrorBanner = document.getElementById('upload-error-banner');
    const processBtn = document.getElementById('process-btn');

    let detectedImagesCount = 0;
    let csvFilesFound = [];
    let rootFolderName = 'Uploaded Dataset';

    files.forEach(file => {
        const path = (file.webkitRelativePath || file.name).replace(/\\/g, '/').toLowerCase();
        const pathParts = path.split('/');
        if (pathParts.length > 1 && pathParts[0]) {
            rootFolderName = file.webkitRelativePath.split('/')[0];
        }

        const isImageExt = /\.(png|jpe?g|bmp|webp)$/i.test(path);
        const isCsvExt = path.endsWith('.csv');

        const isInImagesFolder = path.includes('/images/') || path.startsWith('images/');
        const isInCsvFolder = path.includes('/csv/') || path.startsWith('csv/');

        if ((isInImagesFolder && isImageExt) || (isImageExt && pathParts.length === 1)) {
            detectedImagesCount++;
        }
        if ((isInCsvFolder && isCsvExt) || (isCsvExt && pathParts.length === 1)) {
            csvFilesFound.push(file.name);
        }
    });

    if (detectedImagesCount === 0 && files.filter(f => /\.(png|jpe?g|bmp|webp)$/i.test(f.name)).length > 0) {
        detectedImagesCount = files.filter(f => /\.(png|jpe?g|bmp|webp)$/i.test(f.name)).length;
    }

    selectedFiles = files;

    if (uploadIdleState) uploadIdleState.classList.add('hidden');
    if (uploadStatusCard) uploadStatusCard.classList.remove('hidden');

    const folderDisplay = document.getElementById('folder-name-display');
    if (folderDisplay) folderDisplay.textContent = rootFolderName;

    const imgCountEl = document.getElementById('text-images-count');
    if (imgCountEl) imgCountEl.textContent = `${detectedImagesCount} image file(s)`;

    const csvNameEl = document.getElementById('text-csv-name');
    if (csvNameEl && csvFilesFound.length > 0) csvNameEl.textContent = csvFilesFound[0];

    const imageInfo = document.getElementById('image-info');
    if (imageInfo) imageInfo.textContent = `${detectedImagesCount} image file(s) ready`;

    const statImages = document.getElementById('stat-images');
    if (statImages) statImages.textContent = `${detectedImagesCount} images`;

    if (uploadErrorBanner) uploadErrorBanner.classList.add('hidden');

    if (processBtn) {
        processBtn.disabled = false;
        processBtn.classList.remove('cursor-not-allowed');
    }
}

function resetUploadZone(e) {
    if (e) e.stopPropagation();
    selectedFiles = [];
    const uploadIdleState = document.getElementById('upload-idle-state');
    const uploadStatusCard = document.getElementById('upload-status-card');
    const uploadErrorBanner = document.getElementById('upload-error-banner');
    const processBtn = document.getElementById('process-btn');

    if (uploadIdleState) uploadIdleState.classList.remove('hidden');
    if (uploadStatusCard) uploadStatusCard.classList.add('hidden');
    if (uploadErrorBanner) uploadErrorBanner.classList.add('hidden');

    if (processBtn) {
        processBtn.disabled = true;
        processBtn.classList.add('cursor-not-allowed');
    }
    const imageInfo = document.getElementById('image-info');
    if (imageInfo) imageInfo.textContent = 'No images selected';
}

// ── Bulk Processing ────────────────────────────────────────────────
async function processImages() {
    const processBtn = document.getElementById('process-btn');
    if (selectedFiles.length === 0 || processBtn.disabled) return;

    dualLogsMap = {};
    const logTableBody = document.getElementById('log-table-body');
    const uploadZone = document.getElementById('upload-zone');
    const imageResultsContainer = document.getElementById('image-results-container');

    if (logTableBody) logTableBody.innerHTML = '';
    if (uploadZone) uploadZone.classList.add('hidden');
    if (imageResultsContainer) {
        imageResultsContainer.innerHTML = '';
        imageResultsContainer.classList.remove('hidden');
    }

    processBtn.disabled = true;
    processBtn.textContent = 'Processing Dataset...';
    document.getElementById('progress-status').textContent = 'Uploading and processing dataset...';

    // Start live client-side elapsed timer & animated progress bar
    const processStartTime = performance.now();
    const progressBox = document.getElementById('progress-box');
    const progressBarFill = document.getElementById('progress-bar-fill');
    const progressPercent = document.getElementById('progress-percent');

    if (progressBox) progressBox.classList.remove('opacity-60');

    if (window.liveTimerInterval) clearInterval(window.liveTimerInterval);
    window.liveTimerInterval = setInterval(() => {
        const elapsedMs = performance.now() - processStartTime;
        const elapsedSec = Math.floor(elapsedMs / 1000);
        const m = Math.floor(elapsedSec / 60);
        const s = elapsedSec % 60;
        const timeStr = m > 0 ? `${m}m ${s}s` : `${s}s`;

        const el1 = document.getElementById('stat-elapsed');
        const el2 = document.getElementById('summary-total-time');
        if (el1) el1.textContent = timeStr;
        if (el2) el2.textContent = timeStr;

        // Smoothly animate progress bar up to 95% while waiting for backend
        const simulatedPercent = Math.min(95, Math.floor(100 * (1 - Math.exp(-elapsedMs / 4000))));
        if (progressBarFill) progressBarFill.style.width = `${simulatedPercent}%`;
        if (progressPercent) progressPercent.textContent = `${simulatedPercent}%`;
    }, 150);

    const formData = new FormData();
    selectedFiles.forEach(file => formData.append('files', file, file.webkitRelativePath || file.name));

    try {
        const response = await fetch('/api/process-images', { method: 'POST', body: formData });
        const data = await response.json();
        if (!response.ok) throw new Error(data.error || 'Processing failed');

        if (window.liveTimerInterval) {
            clearInterval(window.liveTimerInterval);
            window.liveTimerInterval = null;
        }

        if (progressBarFill) progressBarFill.style.width = '100%';
        if (progressPercent) progressPercent.textContent = '100%';

        document.getElementById('progress-status').textContent = 'Processing Finished';
        if (data.elapsed) {
            if (document.getElementById('stat-elapsed')) document.getElementById('stat-elapsed').textContent = data.elapsed;
            if (document.getElementById('summary-total-time')) document.getElementById('summary-total-time').textContent = data.elapsed;
        }

        let globalSeqId = 1;
        if (data.results) {
            data.results.forEach((res, i) => {
                const fname = res.original_filename || `Image_${i+1}`;
                const fnameNoExt = fname.replace(/\.[^/.]+$/, "");
                const gtVal = res.expected_gt || (data.gt_mapping && (data.gt_mapping[fname] || data.gt_mapping[fnameNoExt])) || '--';

                // Render side-by-side card
                const card = document.createElement('div');
                card.className = 'bg-zinc-950 rounded border border-zinc-800 p-3 flex flex-col gap-2 transition-colors';
                card.innerHTML = `
                    <div class="flex items-center justify-between px-2.5 py-1 bg-zinc-900/80 rounded border border-zinc-800 text-xs">
                        <span class="font-medium text-zinc-300">${fname}</span>
                    </div>
                    <div class="grid grid-cols-2 gap-4">
                        <div class="flex flex-col items-center gap-1.5">
                            <span class="text-[10px] font-mono text-sky-400 font-semibold">EasyOCR</span>
                            <img src="${res.easyocr_annotated_url}" class="max-h-[300px] object-contain rounded border border-zinc-900">
                        </div>
                        <div class="flex flex-col items-center gap-1.5">
                            <span class="text-[10px] font-mono text-indigo-400 font-semibold">PyTesseract</span>
                            <img src="${res.pytesseract_annotated_url}" class="max-h-[300px] object-contain rounded border border-zinc-900">
                        </div>
                    </div>
                `;
                if (imageResultsContainer) imageResultsContainer.appendChild(card);

                (res.detections || []).forEach(det => {
                    const seqId = globalSeqId++;
                    const key = `${fname}_${seqId}`;
                    const modelEasy = det.easyocr || {};
                    const modelTess = det.pytesseract || {};

                    dualLogsMap[key] = {
                        track_id: seqId,
                        file_name: fname,
                        easy: {
                            track_id: seqId,
                            file_name: fname,
                            matched_gt: gtVal,
                            vehicle_type: det.vehicle_type,
                            color: det.color,
                            plate_number: modelEasy.plate_text,
                            confidence: modelEasy.conf,
                            snapshot_url: modelEasy.snapshot_url,
                            plate_crop_url: modelEasy.crop_url
                        },
                        tess: {
                            track_id: seqId,
                            file_name: fname,
                            matched_gt: gtVal,
                            vehicle_type: det.vehicle_type,
                            color: det.color,
                            plate_number: modelTess.plate_text,
                            confidence: modelTess.conf,
                            snapshot_url: modelTess.snapshot_url,
                            plate_crop_url: modelTess.crop_url
                        }
                    };
                });
            });
        }

        if (data.discarded_stats) renderDiscardedStats(data.discarded_stats);
        if (data.metrics) updateMetricsCards(data.metrics);

        try {
            const cerRes = await fetch('/api/cer-summary');
            if (cerRes.ok) {
                const cerData = await cerRes.json();
                if (cerData.easyocr || cerData.pytesseract) updateMetricsCards(cerData);
            }
        } catch(e) {}

        renderDualTable();

    } catch (err) {
        console.error(err);
        document.getElementById('progress-status').textContent = "Error: " + err.message;
    } finally {
        if (window.liveTimerInterval) {
            clearInterval(window.liveTimerInterval);
            window.liveTimerInterval = null;
        }
        processBtn.disabled = false;
        processBtn.textContent = 'Process & Benchmark Dataset';
    }
}

// ── Metric Cards & Comparison Table ───────────────────────────────
function levenshteinDistance(a, b) {
    if (a.length === 0) return b.length;
    if (b.length === 0) return a.length;
    const matrix = [];
    for (let i = 0; i <= b.length; i++) matrix[i] = [i];
    for (let j = 0; j <= a.length; j++) matrix[0][j] = j;
    for (let i = 1; i <= b.length; i++) {
        for (let j = 1; j <= a.length; j++) {
            matrix[i][j] = b.charAt(i - 1) === a.charAt(j - 1)
                ? matrix[i - 1][j - 1]
                : Math.min(matrix[i - 1][j - 1] + 1, Math.min(matrix[i][j - 1] + 1, matrix[i - 1][j] + 1));
        }
    }
    return matrix[b.length][a.length];
}

function renderDualTable() {
    const logTableBody = document.getElementById('log-table-body');
    const logCountEl = document.getElementById('log-count');
    if (!logTableBody) return;
    logTableBody.innerHTML = '';
    const keys = Object.keys(dualLogsMap);
    if (logCountEl) logCountEl.textContent = `${keys.length} Total`;

    if (keys.length === 0) {
        const tr = document.createElement('tr');
        tr.innerHTML = `<td colspan="7" class="py-8 text-center text-zinc-600 font-mono text-xs">No vehicle detections found in dataset.</td>`;
        logTableBody.appendChild(tr);
        return;
    }

    keys.forEach(key => {
        const pair = dualLogsMap[key];
        const easy = pair.easy || {};
        const tess = pair.tess || {};
        const tr = document.createElement('tr');
        tr.className = "border-b border-zinc-800/30 hover:bg-zinc-900/70 cursor-pointer transition-colors group";
        tr.onclick = () => openDetailModal(key);

        const gtStr = easy.matched_gt || tess.matched_gt || '--';
        const easyText = easy.plate_number ? `${easy.plate_number} (${Math.round(easy.confidence * 100)}%)` : '--';
        const tessText = tess.plate_number ? `${tess.plate_number} (${Math.round(tess.confidence * 100)}%)` : '--';

        const eNorm = (easy.plate_number || '').replace(/\s+/g, '').toUpperCase();
        const tNorm = (tess.plate_number || '').replace(/\s+/g, '').toUpperCase();
        const gtNorm = (gtStr || '').replace(/\s+/g, '').toUpperCase();

        let badgeHtml = '<span class="text-zinc-600 font-mono text-[10px]">--</span>';
        if (gtNorm && gtNorm !== '--') {
            const eMatch = eNorm === gtNorm;
            const tMatch = tNorm === gtNorm;
            if (eMatch && tMatch) {
                badgeHtml = '<span class="px-1.5 py-0.5 rounded text-[10px] font-mono font-medium bg-emerald-950/40 text-emerald-400 border border-emerald-800/50">Match</span>';
            } else if (eMatch) {
                badgeHtml = '<span class="px-1.5 py-0.5 rounded text-[10px] font-mono font-medium bg-sky-950/40 text-sky-400 border border-sky-800/50">EasyOCR</span>';
            } else if (tMatch) {
                badgeHtml = '<span class="px-1.5 py-0.5 rounded text-[10px] font-mono font-medium bg-indigo-950/40 text-indigo-400 border border-indigo-800/50">PyTesseract</span>';
            } else {
                const eDist = eNorm ? levenshteinDistance(eNorm, gtNorm) : 99;
                const tDist = tNorm ? levenshteinDistance(tNorm, gtNorm) : 99;
                if (eDist <= 2 && eDist <= tDist) {
                    badgeHtml = `<span class="px-1.5 py-0.5 rounded text-[10px] font-mono font-medium bg-sky-950/40 text-sky-400/80 border border-sky-900/40">EasyOCR (${eDist} diff)</span>`;
                } else if (tDist <= 2 && tDist < eDist) {
                    badgeHtml = `<span class="px-1.5 py-0.5 rounded text-[10px] font-mono font-medium bg-indigo-950/40 text-indigo-400/80 border border-indigo-900/40">PyTesseract (${tDist} diff)</span>`;
                } else {
                    badgeHtml = '<span class="text-zinc-600 font-mono text-[10px]">Misread</span>';
                }
            }
        }

        tr.innerHTML = `
            <td class="py-2.5 px-2 font-mono text-zinc-400">#${pair.track_id}</td>
            <td class="py-2.5 px-2 font-mono text-zinc-300 truncate max-w-[150px]" title="${pair.file_name}">${pair.file_name}</td>
            <td class="py-2.5 px-2 font-sans font-medium capitalize text-zinc-300">${easy.vehicle_type || 'Vehicle'} (${easy.color || ''})</td>
            <td class="py-2.5 px-2 font-semibold text-sky-400 group-hover:underline">${easyText}</td>
            <td class="py-2.5 px-2 font-semibold text-indigo-400 group-hover:underline">${tessText}</td>
            <td class="py-2.5 px-2 text-zinc-300 font-mono font-bold">${gtStr}</td>
            <td class="py-2.5 px-2 flex items-center gap-2">${badgeHtml}<span class="text-[10px] text-zinc-500 group-hover:text-zinc-300 underline font-sans ml-auto">Details →</span></td>
        `;
        logTableBody.appendChild(tr);
    });
}

// ── Detail Telemetry Modal ─────────────────────────────────────────
function openDetailModal(key) {
    const pair = dualLogsMap[key];
    if (!pair) return;
    const easy = pair.easy || {};
    const tess = pair.tess || {};

    document.getElementById('modal-track-id').textContent = `#${pair.track_id || '--'}`;
    document.getElementById('modal-meta').textContent = `${(easy.vehicle_type || 'VEHICLE').toUpperCase()} ${(easy.color || '').toUpperCase()} • File: ${pair.file_name} • Matched GT: ${easy.matched_gt || '--'}`;

    let snapUrl = easy.snapshot_url || tess.snapshot_url;
    if (snapUrl && !snapUrl.startsWith('/') && !snapUrl.startsWith('http')) snapUrl = '/' + snapUrl;
    const snapImg = document.getElementById('modal-snapshot-img');
    const snapPlaceholder = document.getElementById('modal-snapshot-placeholder');
    if (snapUrl && snapImg) {
        snapImg.src = snapUrl;
        snapImg.classList.remove('hidden');
        if (snapPlaceholder) snapPlaceholder.classList.add('hidden');
    } else if (snapPlaceholder) {
        snapImg.classList.add('hidden');
        snapPlaceholder.classList.remove('hidden');
    }

    if (document.getElementById('modal-easy-read')) document.getElementById('modal-easy-read').textContent = easy.plate_number || 'No Detection';
    if (document.getElementById('modal-easy-conf')) document.getElementById('modal-easy-conf').textContent = easy.confidence ? `${(easy.confidence * 100).toFixed(0)}% conf` : '0% conf';
    if (document.getElementById('modal-easy-cer')) document.getElementById('modal-easy-cer').textContent = easy.cer != null ? `${(easy.cer * 100).toFixed(1)}%` : '--';

    const easyCropImg = document.getElementById('modal-easy-crop');
    const easyCropEmpty = document.getElementById('modal-easy-crop-empty');
    let easyCropSrc = easy.plate_crop_url || tess.plate_crop_url;
    if (easyCropSrc && !easyCropSrc.startsWith('/') && !easyCropSrc.startsWith('http')) easyCropSrc = '/' + easyCropSrc;
    if (easyCropSrc && easyCropImg) {
        easyCropImg.src = easyCropSrc;
        easyCropImg.classList.remove('hidden');
        if (easyCropEmpty) easyCropEmpty.classList.add('hidden');
    } else if (easyCropEmpty) {
        easyCropImg.classList.add('hidden');
        easyCropEmpty.classList.remove('hidden');
    }

    if (document.getElementById('modal-tess-read')) document.getElementById('modal-tess-read').textContent = tess.plate_number || 'No Detection';
    if (document.getElementById('modal-tess-conf')) document.getElementById('modal-tess-conf').textContent = tess.confidence ? `${(tess.confidence * 100).toFixed(0)}% conf` : '0% conf';
    if (document.getElementById('modal-tess-cer')) document.getElementById('modal-tess-cer').textContent = tess.cer != null ? `${(tess.cer * 100).toFixed(1)}%` : '--';

    const tessCropImg = document.getElementById('modal-tess-crop');
    const tessCropEmpty = document.getElementById('modal-tess-crop-empty');
    let tessCropSrc = tess.plate_crop_url || easy.plate_crop_url;
    if (tessCropSrc && !tessCropSrc.startsWith('/') && !tessCropSrc.startsWith('http')) tessCropSrc = '/' + tessCropSrc;
    if (tessCropSrc && tessCropImg) {
        tessCropImg.src = tessCropSrc;
        tessCropImg.classList.remove('hidden');
        if (tessCropEmpty) tessCropEmpty.classList.add('hidden');
    } else if (tessCropEmpty) {
        tessCropImg.classList.add('hidden');
        tessCropEmpty.classList.remove('hidden');
    }

    const modal = document.getElementById('detail-modal');
    if (modal) modal.classList.remove('hidden');
}

function closeDetailModal() {
    const modal = document.getElementById('detail-modal');
    if (modal) modal.classList.add('hidden');
}

function updateMetricsCards(metrics) {
    if (!metrics) return;
    const easy = metrics.easyocr || {};
    const tess = metrics.pytesseract || {};

    if (document.getElementById('easy-ema-val')) document.getElementById('easy-ema-val').textContent = (easy.exact_match_rate != null ? (easy.exact_match_rate * 100).toFixed(1) : '--') + '%';
    if (document.getElementById('easy-ha-val')) document.getElementById('easy-ha-val').textContent = (easy.high_accuracy_rate != null ? (easy.high_accuracy_rate * 100).toFixed(1) : '--') + '%';
    if (document.getElementById('easy-crr-val')) document.getElementById('easy-crr-val').textContent = (easy.crr != null ? easy.crr.toFixed(1) : '--') + '%';
    if (document.getElementById('easy-lat-val')) document.getElementById('easy-lat-val').textContent = (easy.latency_per_plate_ms != null ? easy.latency_per_plate_ms.toFixed(0) : '--') + 'ms';
    if (document.getElementById('easy-ema-badge')) document.getElementById('easy-ema-badge').textContent = 'EMA: ' + (easy.exact_match_rate != null ? (easy.exact_match_rate * 100).toFixed(1) : '--') + '%';

    if (document.getElementById('tess-ema-val')) document.getElementById('tess-ema-val').textContent = (tess.exact_match_rate != null ? (tess.exact_match_rate * 100).toFixed(1) : '--') + '%';
    if (document.getElementById('tess-ha-val')) document.getElementById('tess-ha-val').textContent = (tess.high_accuracy_rate != null ? (tess.high_accuracy_rate * 100).toFixed(1) : '--') + '%';
    if (document.getElementById('tess-crr-val')) document.getElementById('tess-crr-val').textContent = (tess.crr != null ? tess.crr.toFixed(1) : '--') + '%';
    if (document.getElementById('tess-lat-val')) document.getElementById('tess-lat-val').textContent = (tess.latency_per_plate_ms != null ? tess.latency_per_plate_ms.toFixed(0) : '--') + 'ms';
    if (document.getElementById('tess-ema-badge')) document.getElementById('tess-ema-badge').textContent = 'EMA: ' + (tess.exact_match_rate != null ? (tess.exact_match_rate * 100).toFixed(1) : '--') + '%';

    if (metrics.winner && document.getElementById('stat-winner-name')) {
        document.getElementById('stat-winner-name').textContent = metrics.winner;
    }

    if (metrics.chart_url && document.getElementById('matplotlib-img')) {
        document.getElementById('matplotlib-img').src = metrics.chart_url + '?t=' + Date.now();
        document.getElementById('matplotlib-img').classList.remove('hidden');
        if (document.getElementById('matplotlib-placeholder')) {
            document.getElementById('matplotlib-placeholder').classList.add('hidden');
        }
    }
}

function renderDiscardedStats(stats) {
    const sec = document.getElementById('discarded-section');
    const toggleBtn = document.getElementById('toggle-discarded-btn');
    if (!sec) return;

    if (!stats || !stats.total_discarded || stats.total_discarded === 0) {
        sec.classList.add('hidden');
        if (toggleBtn) toggleBtn.classList.add('hidden');
        return;
    }

    sec.classList.remove('hidden');
    if (document.getElementById('discarded-total-val')) document.getElementById('discarded-total-val').textContent = stats.total_discarded;
    if (document.getElementById('discarded-nocar-val')) document.getElementById('discarded-nocar-val').textContent = stats.no_vehicle_count;
    if (document.getElementById('discarded-noplate-val')) document.getElementById('discarded-noplate-val').textContent = stats.no_plate_count;
    if (document.getElementById('discarded-badge')) document.getElementById('discarded-badge').textContent = `${stats.total_discarded} Discarded`;

    if (toggleBtn) {
        if (stats.discarded_files && stats.discarded_files.length > 0) {
            toggleBtn.classList.remove('hidden');
        } else {
            toggleBtn.classList.add('hidden');
        }
    }
}

async function fetchCerSummary() {
    try {
        const res = await fetch('/api/cer-summary');
        const data = await res.json();
        if (data.easyocr || data.pytesseract) updateMetricsCards(data);
    } catch (e) { console.warn(e); }
}

async function loadGroundTruth() {
    try {
        const res = await fetch('/api/ground-truth');
        const data = await res.json();
        if (document.getElementById('gt-count')) document.getElementById('gt-count').textContent = (data.plates || []).length + ' plates';
    } catch (e) { console.warn(e); }
}
