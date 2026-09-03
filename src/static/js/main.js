// ── Global State Variables ──────────────────────────────────────────
let selectedFiles = [];
let dualLogsMap = {};
let lastMetrics = null;

// Table Pagination & Filtering State
let currentPage = 1;
const pageSize = 20;
let currentFilter = 'all';
let currentSort = 'easy_conf_desc';

// ── DOM Initialization ─────────────────────────────────────────────
document.addEventListener('DOMContentLoaded', () => {
    const uploadZone = document.getElementById('upload-zone');
    const fileInput = document.getElementById('image-file-input') || document.getElementById('file-input');
    const singleFileInput = document.getElementById('single-file-input');
    const processBtn = document.getElementById('process-btn');
    const uploadIdleState = document.getElementById('upload-idle-state');
    const uploadStatusCard = document.getElementById('upload-status-card');
    const uploadErrorBanner = document.getElementById('upload-error-banner');

    const filterSelect = document.getElementById('table-filter');
    const sortSelect = document.getElementById('table-sort');
    const prevBtn = document.getElementById('btn-prev-page');
    const nextBtn = document.getElementById('btn-next-page');

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

    if (filterSelect) {
        filterSelect.addEventListener('change', (e) => {
            currentFilter = e.target.value;
            currentPage = 1;
            renderDualTable();
        });
    }

    if (sortSelect) {
        sortSelect.addEventListener('change', (e) => {
            currentSort = e.target.value;
            currentPage = 1;
            renderDualTable();
        });
    }

    if (prevBtn) {
        prevBtn.addEventListener('click', () => {
            if (currentPage > 1) {
                currentPage--;
                renderDualTable();
            }
        });
    }

    if (nextBtn) {
        nextBtn.addEventListener('click', () => {
            currentPage++;
            renderDualTable();
        });
    }

    const pageInput = document.getElementById('page-input');
    if (pageInput) {
        const handlePageChange = (e) => {
            const val = parseInt(e.target.value, 10);
            if (!isNaN(val)) {
                currentPage = val;
                renderDualTable();
            }
        };
        pageInput.addEventListener('change', handlePageChange);
        pageInput.addEventListener('keydown', (e) => {
            if (e.key === 'Enter') handlePageChange(e);
        });
    }

    window.addEventListener('keydown', (e) => {
        if (e.key === 'Escape') closeDetailModal();
    });

    initThemeToggle();
    loadGroundTruth();
});

// ── Dark / Light Mode Theme Controller ─────────────────────────────
function initThemeToggle() {
    const themeToggleBtn = document.getElementById('theme-toggle');
    const themeToggleText = document.getElementById('theme-toggle-text');

    function updateThemeUI(isDark) {
        if (themeToggleText) {
            themeToggleText.textContent = isDark ? 'Dark Mode' : 'Light Mode';
        }
    }

    const isInitialDark = document.documentElement.classList.contains('dark');
    updateThemeUI(isInitialDark);

    if (themeToggleBtn) {
        themeToggleBtn.addEventListener('click', () => {
            // Apply synchronized smooth transition to all elements across the entire page
            document.documentElement.classList.add('theme-transition');

            const isDark = document.documentElement.classList.toggle('dark');
            localStorage.setItem('theme', isDark ? 'dark' : 'light');
            updateThemeUI(isDark);

            // Clean up transition class after transition completes (350ms)
            setTimeout(() => {
                document.documentElement.classList.remove('theme-transition');
            }, 350);
        });
    }
}

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
    if (imgCountEl) imgCountEl.textContent = `${detectedImagesCount} image files`;

    const csvNameEl = document.getElementById('text-csv-name');
    if (csvNameEl) {
        if (csvFilesFound.length > 0) {
            csvNameEl.textContent = csvFilesFound[0];
            csvNameEl.className = 'text-emerald-400 font-medium truncate';
        } else {
            csvNameEl.textContent = 'No CSV ground truth';
            csvNameEl.className = 'text-rose-400 font-medium truncate';
        }
    }

    const imageInfo = document.getElementById('image-info');
    if (imageInfo) imageInfo.textContent = `${detectedImagesCount} image files ready`;

    const statImages = document.getElementById('stat-images');
    if (statImages) statImages.textContent = `${detectedImagesCount} images`;

    const uploadErrorText = document.getElementById('upload-error-text');
    const isValid = detectedImagesCount > 0 && csvFilesFound.length > 0;

    if (!isValid) {
        let errMessage = 'Invalid dataset structure.';
        if (detectedImagesCount === 0 && csvFilesFound.length === 0) {
            errMessage = 'No image files or CSV ground truth found in selected folder.';
        } else if (detectedImagesCount === 0) {
            errMessage = 'No image files found. Folder must contain an \'images/\' subfolder or image files.';
        } else if (csvFilesFound.length === 0) {
            errMessage = 'No CSV ground truth file found. Folder must contain a \'csv/\' subfolder or a .csv file.';
        }
        if (uploadErrorText) uploadErrorText.textContent = errMessage;
        if (uploadErrorBanner) uploadErrorBanner.classList.remove('hidden');
        if (processBtn) {
            processBtn.disabled = true;
            processBtn.classList.add('cursor-not-allowed');
        }
    } else {
        if (uploadErrorBanner) uploadErrorBanner.classList.add('hidden');
        if (processBtn) {
            processBtn.disabled = false;
            processBtn.classList.remove('cursor-not-allowed');
        }
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
    currentPage = 1;
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

        const totalWallMs = performance.now() - processStartTime;
        const totalWallSec = Math.max(1, Math.round(totalWallMs / 1000));
        const finalMinutes = Math.floor(totalWallSec / 60);
        const finalSeconds = totalWallSec % 60;
        const totalWallFormatted = finalMinutes > 0 ? `${finalMinutes}m ${finalSeconds}s` : `${finalSeconds}s`;

        if (window.liveTimerInterval) {
            clearInterval(window.liveTimerInterval);
            window.liveTimerInterval = null;
        }

        if (progressBarFill) progressBarFill.style.width = '100%';
        if (progressPercent) progressPercent.textContent = '100%';

        document.getElementById('progress-status').textContent = 'Processing Finished';
        if (document.getElementById('stat-elapsed')) document.getElementById('stat-elapsed').textContent = totalWallFormatted;
        if (document.getElementById('summary-total-time')) document.getElementById('summary-total-time').textContent = totalWallFormatted;

        let globalSeqId = 1;
        if (data.results) {
            data.results.forEach((res, i) => {
                const fname = res.original_filename || `Image_${i+1}`;
                const fnameNoExt = fname.replace(/\.[^/.]+$/, "");
                const gtVal = res.expected_gt || (data.gt_mapping && (data.gt_mapping[fname] || data.gt_mapping[fnameNoExt])) || '--';

                // Render side-by-side card
                const card = document.createElement('div');
                card.className = 'bg-white dark:bg-zinc-950 rounded-xl border border-zinc-200 dark:border-zinc-800 p-3.5 flex flex-col gap-2 transition-colors shadow-sm';
                card.innerHTML = `
                    <div class="flex items-center justify-between px-3 py-1.5 bg-zinc-50 dark:bg-zinc-900/80 rounded-lg border border-zinc-200 dark:border-zinc-800 text-xs">
                        <span class="font-medium text-zinc-800 dark:text-zinc-200 font-mono">${fname}</span>
                    </div>
                    <div class="grid grid-cols-2 gap-4">
                        <div class="flex flex-col items-center gap-1.5">
                            <span class="text-[10px] font-mono text-sky-600 dark:text-sky-400 font-semibold">EasyOCR</span>
                            <img src="${res.easyocr_annotated_url}" class="max-h-[300px] object-contain rounded-lg border border-zinc-200 dark:border-zinc-900 shadow-sm">
                        </div>
                        <div class="flex flex-col items-center gap-1.5">
                            <span class="text-[10px] font-mono text-indigo-600 dark:text-indigo-400 font-semibold">PyTesseract</span>
                            <img src="${res.pytesseract_annotated_url}" class="max-h-[300px] object-contain rounded-lg border border-zinc-200 dark:border-zinc-900 shadow-sm">
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

function updateMetricsCards(metrics) {
    if (!metrics) return;

    const winnerName = document.getElementById('stat-winner-name');
    if (winnerName) winnerName.textContent = metrics.winner || 'Tie';

    const matImg = document.getElementById('matplotlib-img');
    const matPlaceholder = document.getElementById('matplotlib-placeholder');
    if (metrics.chart_url && matImg) {
        matImg.src = metrics.chart_url + '?t=' + Date.now();
        matImg.classList.remove('hidden');
        if (matPlaceholder) matPlaceholder.classList.add('hidden');
    }
}

function renderDualTable() {
    const logTableBody = document.getElementById('log-table-body');
    const logCountEl = document.getElementById('log-count');
    const paginationInfoEl = document.getElementById('pagination-info');
    const pageInput = document.getElementById('page-input');
    const pageTotalEl = document.getElementById('page-total');
    const prevBtn = document.getElementById('btn-prev-page');
    const nextBtn = document.getElementById('btn-next-page');

    if (!logTableBody) return;
    logTableBody.innerHTML = '';

    const keys = Object.keys(dualLogsMap);
    if (logCountEl) logCountEl.textContent = `${keys.length} Total`;

    if (keys.length === 0) {
        const tr = document.createElement('tr');
        tr.className = "h-12";
        tr.innerHTML = `<td colspan="7" class="py-4 text-center text-zinc-400 dark:text-zinc-600 font-mono text-xs align-middle">No vehicle detections found in dataset.</td>`;
        logTableBody.appendChild(tr);
        if (paginationInfoEl) paginationInfoEl.textContent = 'Showing 0 of 0 records';
        if (pageInput) { pageInput.value = 1; pageInput.max = 1; }
        if (pageTotalEl) pageTotalEl.textContent = '1';
        if (prevBtn) prevBtn.disabled = true;
        if (nextBtn) nextBtn.disabled = true;
        return;
    }

    let records = keys.map(key => ({ key, ...dualLogsMap[key] }));

    function getStatus(pair) {
        const easy = pair.easy || {};
        const tess = pair.tess || {};
        const gtStr = easy.matched_gt || tess.matched_gt || '--';
        const eNorm = (easy.plate_number || '').replace(/\s+/g, '').toUpperCase();
        const tNorm = (tess.plate_number || '').replace(/\s+/g, '').toUpperCase();
        const gtNorm = (gtStr || '').replace(/\s+/g, '').toUpperCase();

        if (!gtNorm || gtNorm === '--') return 'unknown';

        const eMatch = eNorm === gtNorm;
        const tMatch = tNorm === gtNorm;

        if (eMatch && tMatch) return 'both_match';
        if (eMatch) return 'easy_match';
        if (tMatch) return 'tess_match';
        return 'misread';
    }

    // 1. Filtering
    if (currentFilter === 'hide_misreads') {
        records = records.filter(r => ['both_match', 'easy_match', 'tess_match'].includes(getStatus(r)));
    } else if (currentFilter === 'misreads_only') {
        records = records.filter(r => getStatus(r) === 'misread');
    } else if (currentFilter === 'easy_match') {
        records = records.filter(r => ['both_match', 'easy_match'].includes(getStatus(r)));
    } else if (currentFilter === 'tess_match') {
        records = records.filter(r => ['both_match', 'tess_match'].includes(getStatus(r)));
    }

    // 2. Sorting (by OCR Confidence)
    records.sort((a, b) => {
        if (currentSort === 'tess_conf_desc') {
            return (b.tess?.confidence || 0) - (a.tess?.confidence || 0);
        }
        // Default: EasyOCR Confidence (High -> Low)
        return (b.easy?.confidence || 0) - (a.easy?.confidence || 0);
    });

    const totalFiltered = records.length;
    const totalPages = Math.max(1, Math.ceil(totalFiltered / pageSize));

    if (currentPage > totalPages) currentPage = totalPages;
    if (currentPage < 1) currentPage = 1;

    const startIdx = (currentPage - 1) * pageSize;
    const endIdx = Math.min(startIdx + pageSize, totalFiltered);
    const pageRecords = records.slice(startIdx, endIdx);

    // Update Pagination Controls UI
    if (paginationInfoEl) {
        paginationInfoEl.textContent = totalFiltered > 0
            ? `Showing ${startIdx + 1}-${endIdx} of ${totalFiltered} records`
            : `Showing 0 of 0 records`;
    }
    if (pageInput) {
        pageInput.value = currentPage;
        pageInput.max = totalPages;
    }
    if (pageTotalEl) {
        pageTotalEl.textContent = totalPages;
    }
    if (prevBtn) prevBtn.disabled = currentPage <= 1;
    if (nextBtn) nextBtn.disabled = currentPage >= totalPages;

    if (pageRecords.length === 0) {
        const tr = document.createElement('tr');
        tr.className = "h-12";
        tr.innerHTML = `<td colspan="7" class="py-4 text-center text-zinc-400 dark:text-zinc-600 font-mono text-xs align-middle">No records match the selected filter.</td>`;
        logTableBody.appendChild(tr);
        return;
    }

    pageRecords.forEach(pair => {
        const key = pair.key;
        const easy = pair.easy || {};
        const tess = pair.tess || {};
        const tr = document.createElement('tr');
        tr.className = "border-b border-zinc-200 dark:border-zinc-800/30 hover:bg-zinc-50 dark:hover:bg-zinc-900/70 cursor-pointer transition-colors group h-11";
        tr.onclick = () => openDetailModal(key);

        const gtStr = easy.matched_gt || tess.matched_gt || '--';
        const easyText = easy.plate_number ? `${easy.plate_number} (${Math.round((easy.confidence || 0) * 100)}%)` : '--';
        const tessText = tess.plate_number ? `${tess.plate_number} (${Math.round((tess.confidence || 0) * 100)}%)` : '--';

        const status = getStatus(pair);
        let badgeHtml = '<span class="text-zinc-400 dark:text-zinc-600 font-mono text-[10px] inline-block">--</span>';
        if (status === 'both_match') {
            badgeHtml = '<span class="px-1.5 py-0.5 rounded text-[10px] font-mono font-medium bg-emerald-50 dark:bg-emerald-950/40 text-emerald-700 dark:text-emerald-400 border border-emerald-200 dark:border-emerald-800/50 inline-block leading-none">Match</span>';
        } else if (status === 'easy_match') {
            badgeHtml = '<span class="px-1.5 py-0.5 rounded text-[10px] font-mono font-medium bg-sky-50 dark:bg-sky-950/40 text-sky-700 dark:text-sky-400 border border-sky-200 dark:border-sky-800/50 inline-block leading-none">EasyOCR</span>';
        } else if (status === 'tess_match') {
            badgeHtml = '<span class="px-1.5 py-0.5 rounded text-[10px] font-mono font-medium bg-indigo-50 dark:bg-indigo-950/40 text-indigo-700 dark:text-indigo-400 border border-indigo-200 dark:border-indigo-800/50 inline-block leading-none">PyTesseract</span>';
        } else if (status === 'misread') {
            badgeHtml = '<span class="text-zinc-400 dark:text-zinc-600 font-mono text-[10px] inline-block">Misread</span>';
        }

        tr.innerHTML = `
            <td class="py-2 px-2.5 font-mono text-zinc-500 dark:text-zinc-400 align-middle whitespace-nowrap">#${pair.track_id}</td>
            <td class="py-2 px-2.5 font-mono text-zinc-800 dark:text-zinc-300 truncate max-w-[150px] align-middle whitespace-nowrap font-medium" title="${pair.file_name}">${pair.file_name}</td>
            <td class="py-2 px-2.5 font-sans font-medium capitalize text-zinc-700 dark:text-zinc-300 align-middle whitespace-nowrap">${easy.vehicle_type || 'Vehicle'}</td>
            <td class="py-2 px-2.5 font-semibold text-sky-600 dark:text-sky-400 group-hover:underline align-middle whitespace-nowrap">${easyText}</td>
            <td class="py-2 px-2.5 font-semibold text-indigo-600 dark:text-indigo-400 group-hover:underline align-middle whitespace-nowrap">${tessText}</td>
            <td class="py-2 px-2.5 text-zinc-900 dark:text-zinc-300 font-mono font-bold align-middle whitespace-nowrap">${gtStr}</td>
            <td class="py-2 px-2.5 align-middle whitespace-nowrap">
                <div class="flex items-center gap-2 h-6">
                    ${badgeHtml}
                    <span class="text-[10px] text-zinc-400 dark:text-zinc-500 group-hover:text-zinc-800 dark:group-hover:text-zinc-300 underline font-sans ml-auto">Details →</span>
                </div>
            </td>
        `;
        logTableBody.appendChild(tr);
    });
}

function _formatImgUrl(url) {
    if (!url) return '';
    if (url.startsWith('data:') || url.startsWith('/') || url.startsWith('http')) return url;
    return '/' + url;
}

// ── Detail Telemetry Modal ─────────────────────────────────────────
function openDetailModal(key) {
    const pair = dualLogsMap[key];
    if (!pair) return;
    const easy = pair.easy || {};
    const tess = pair.tess || {};

    document.getElementById('modal-track-id').textContent = `#${pair.track_id || '--'}`;
    document.getElementById('modal-meta').textContent = `${(easy.vehicle_type || 'VEHICLE').toUpperCase()} • File: ${pair.file_name} • Matched GT: ${easy.matched_gt || '--'}`;

    const snapUrl = _formatImgUrl(easy.snapshot_url || tess.snapshot_url);
    const snapImg = document.getElementById('modal-snapshot-img');
    const snapPlaceholder = document.getElementById('modal-snapshot-placeholder');
    if (snapUrl && snapImg) {
        snapImg.src = snapUrl;
        snapImg.classList.remove('hidden');
        if (snapPlaceholder) snapPlaceholder.classList.add('hidden');
    } else if (snapPlaceholder) {
        if (snapImg) snapImg.classList.add('hidden');
        snapPlaceholder.classList.remove('hidden');
    }

    if (document.getElementById('modal-easy-read')) document.getElementById('modal-easy-read').textContent = easy.plate_number || 'No Detection';
    if (document.getElementById('modal-easy-conf')) document.getElementById('modal-easy-conf').textContent = easy.confidence ? `${(easy.confidence * 100).toFixed(0)}% conf` : '0% conf';
    if (document.getElementById('modal-easy-cer')) document.getElementById('modal-easy-cer').textContent = easy.cer != null ? `${(easy.cer * 100).toFixed(1)}%` : '--';

    const easyCropImg = document.getElementById('modal-easy-crop');
    const easyCropEmpty = document.getElementById('modal-easy-crop-empty');
    const easyCropSrc = _formatImgUrl(easy.plate_crop_url || tess.plate_crop_url);
    if (easyCropSrc && easyCropImg) {
        easyCropImg.src = easyCropSrc;
        easyCropImg.classList.remove('hidden');
        if (easyCropEmpty) easyCropEmpty.classList.add('hidden');
    } else if (easyCropEmpty) {
        if (easyCropImg) easyCropImg.classList.add('hidden');
        easyCropEmpty.classList.remove('hidden');
    }

    if (document.getElementById('modal-tess-read')) document.getElementById('modal-tess-read').textContent = tess.plate_number || 'No Detection';
    if (document.getElementById('modal-tess-conf')) document.getElementById('modal-tess-conf').textContent = tess.confidence ? `${(tess.confidence * 100).toFixed(0)}% conf` : '0% conf';
    if (document.getElementById('modal-tess-cer')) document.getElementById('modal-tess-cer').textContent = tess.cer != null ? `${(tess.cer * 100).toFixed(1)}%` : '--';

    const tessCropImg = document.getElementById('modal-tess-crop');
    const tessCropEmpty = document.getElementById('modal-tess-crop-empty');
    const tessCropSrc = _formatImgUrl(tess.plate_crop_url || easy.plate_crop_url);
    if (tessCropSrc && tessCropImg) {
        tessCropImg.src = tessCropSrc;
        tessCropImg.classList.remove('hidden');
        if (tessCropEmpty) tessCropEmpty.classList.add('hidden');
    } else if (tessCropEmpty) {
        if (tessCropImg) tessCropImg.classList.add('hidden');
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
    if (document.getElementById('easy-crr-val')) document.getElementById('easy-crr-val').textContent = (easy.crr != null ? easy.crr.toFixed(1) : '--') + '%';
    if (document.getElementById('easy-lat-val')) document.getElementById('easy-lat-val').textContent = (easy.latency_per_plate_ms != null ? easy.latency_per_plate_ms.toFixed(0) : '--') + 'ms';
    if (document.getElementById('easy-ema-badge')) document.getElementById('easy-ema-badge').textContent = 'EMA: ' + (easy.exact_match_rate != null ? (easy.exact_match_rate * 100).toFixed(1) : '--') + '%';

    if (document.getElementById('tess-ema-val')) document.getElementById('tess-ema-val').textContent = (tess.exact_match_rate != null ? (tess.exact_match_rate * 100).toFixed(1) : '--') + '%';
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
    if (!sec) return;

    if (!stats || !stats.total_discarded || stats.total_discarded === 0) {
        sec.classList.add('hidden');
        return;
    }

    sec.classList.remove('hidden');
    if (document.getElementById('discarded-total-val')) document.getElementById('discarded-total-val').textContent = stats.total_discarded;
    if (document.getElementById('discarded-nocar-val')) document.getElementById('discarded-nocar-val').textContent = stats.no_vehicle_count;
    if (document.getElementById('discarded-noplate-val')) document.getElementById('discarded-noplate-val').textContent = stats.no_plate_count;
    if (document.getElementById('discarded-badge')) document.getElementById('discarded-badge').textContent = `${stats.total_discarded} Discarded`;
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
