$env:CUDA_VISIBLE_DEVICES = "-1"   # CPU only -- GPU is busy with egwt-reproduction's freeze-mode comparison
$py = "C:\Users\peter\miniconda3\python.exe"
$src = "E:\plant_disease\LightETFusion_repro\src"
$dataRoot = "E:\plant_disease\LightETFusion_repro\data\splits"
$runsDir = "E:\plant_disease\LightETFusion_repro\runs"
Set-Location $src

foreach ($seed in 1,2,3) {
  $splitDir = Join-Path $dataRoot "seed$seed"
  $backboneDir = Join-Path $runsDir "seed$seed"
  $outDir = Join-Path $backboneDir "svmC10"
  $log = Join-Path $runsDir "phase2_svmC10_seed${seed}_log.txt"
  Write-Output "=== seed $seed : SVM C=10.0 ==="
  & $py -u train_phase2.py `
      --data_dir $splitDir `
      --backbone_dir $backboneDir `
      --output_dir $outDir `
      --seed $seed `
      --svm_C 10.0 `
    2>&1 | Tee-Object -FilePath $log
}

Write-Output "=== aggregating 3-seed svmC10 summary ==="
& $py -u aggregate_svmC10.py

Write-Output "=== SVM-C10 RERUN DONE ==="
