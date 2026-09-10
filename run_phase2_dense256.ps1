$env:CUDA_VISIBLE_DEVICES = "-1"   # CPU only -- GPU is busy with egwt-reproduction's ImageNet pretrain
$py = "C:\Users\peter\miniconda3\python.exe"
$src = "E:\plant_disease\LightETFusion_repro\src"
$dataRoot = "E:\plant_disease\LightETFusion_repro\data\splits"
$runsDir = "E:\plant_disease\LightETFusion_repro\runs"
Set-Location $src

foreach ($seed in 1,2,3) {
  $splitDir = Join-Path $dataRoot "seed$seed"
  $backboneDir = Join-Path $runsDir "seed$seed"
  $outDir = Join-Path $backboneDir "dense256"
  $log = Join-Path $runsDir "phase2_dense256_seed${seed}_log.txt"
  Write-Output "=== seed $seed : extracting from Dense(256) instead of Dense(128) ==="
  & $py -u train_phase2.py `
      --data_dir $splitDir `
      --backbone_dir $backboneDir `
      --output_dir $outDir `
      --seed $seed `
      --feature_layer 256 `
    2>&1 | Tee-Object -FilePath $log
}

Write-Output "=== aggregating 3-seed dense256 summary ==="
& $py -u aggregate_dense256.py

Write-Output "=== DENSE256 RERUN DONE ==="
