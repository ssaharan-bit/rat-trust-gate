#!/usr/bin/env bash
# Clone the official code of the published methods at the exact commits used in the paper.
# The repositories are not redistributed here; see their own licences.
set -euo pipefail
EXT="$(cd "$(dirname "$0")/.." && pwd)/rat/ext"
mkdir -p "$EXT"
clone() { [ -d "$EXT/$1" ] || git clone -q "$2" "$EXT/$1"; git -C "$EXT/$1" checkout -q "$3"; echo "$1 @ ${3:0:10}"; }
clone "BaoquanGong_Multimodal-Negative-Learning" "https://github.com/BaoquanGong/Multimodal-Negative-Learning" "64cea5624d5894ade5f087ec8d72623b0adb9f13"
clone "JethroJames_TUNED" "https://github.com/JethroJames/TUNED" "6079196e302e39bf6d0e6c604edd0108f284f8c6"
clone "MOGONET" "https://github.com/txWang/MOGONET.git" "32d9066b7fe289b3f7cdb496a5668834079c274c"
clone "PDF" "https://github.com/Yinan-Xia/PDF" "864867426cde076fb3d0529d4df6b440d6963451"
clone "QMF" "https://github.com/QingyangZhang/QMF.git" "fe6c4c6ef7cb23f0a89594ee413d485f1854268b"
clone "RCML" "https://github.com/jiajunsi/RCML" "c9c5ab41e6fe62a85e5f6441a4dc7b568e1fa421"
clone "TMC" "https://github.com/hanmenghan/TMC" "a3272b8746861c76a3461943b5eee51df5b5a8fe"
clone "XLearning-SCU_2026-CVPR-BML" "https://github.com/XLearning-SCU/2026-CVPR-BML" "49f8652da2a1dd5b97aabca4f745deda62e4ccab"
clone "cncq-tang_GTMC-HOA" "https://github.com/cncq-tang/GTMC-HOA" "39a3a2e4fea2c3c3226a02e2943282b837928d38"
clone "cncq-tang_TMLC" "https://github.com/cncq-tang/TMLC" "b4b43bc9b8c74abd152d78345fd38d43ab2f79e7"
clone "jie019_TMCEK_ICML2025" "https://github.com/jie019/TMCEK_ICML2025" "0e26e9a8f7f8ef7ecdc118bb4878786ee61b3f37"
clone "lshi91_GTMC-HOA" "https://github.com/lshi91/GTMC-HOA" "39a3a2e4fea2c3c3226a02e2943282b837928d38"
clone "mmdynamics" "https://github.com/TencentAILabHealthcare/mmdynamics" "06a2bb8a66953be77c53e6a027bc04193c5bdca4"
clone "shicaiwei123_UDML" "https://github.com/shicaiwei123/UDML" "fc66740b270a43793653cbd8b1fcccc9376af029"
ln -sfn ../rat/ext "$(dirname "$EXT")/../studies/ext"   # study-local copies of the method code use the same checkouts
