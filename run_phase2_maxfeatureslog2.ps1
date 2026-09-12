$env:CUDA_VISIBLE_DEVICES = "-1"   # CPU only -- GPU is busy with ConViTX_repro's ResNet-50/PlantVillage pretrain
$py = "C:\Users\peter\miniconda3\python.exe"
$src = "E:\plant_disease\LightETFusion_repro\src"
$dataRoot = "E:\plant_disease\LightETFusion_repro\data\splits"
$runsDir = "E:\plant_disease\LightETFusion_repro\runs"
Set-Location $src

foreach ($seed in 1,2,3) {
  $splitDir = Join-Path $dataRoot "seed$seed"
  $backboneDir = Join-Path $runsDir "seed$seed"
  $outDir = Join-Path $backboneDir "maxfeatureslog2"
  $log = Join-Path $runsDir "phase2_maxfeatureslog2_seed${seed}_log.txt"
  Write-Output "=== seed $seed : ExtraTrees max_features=log2 ==="
  & $py -u train_phase2.py `
      --data_dir $splitDir `
      --backbone_dir $backboneDir `
      --output_dir $outDir `
      --seed $seed `
      --et_max_features log2 `
    2>&1 | Tee-Object -FilePath $log
}

Write-Output "=== aggregating 3-seed max_features=log2 summary ==="
& $py -u aggregate_maxfeatureslog2.py

Write-Output "=== MAX_FEATURES=LOG2 RERUN DONE ==="
