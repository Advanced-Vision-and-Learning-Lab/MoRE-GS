#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
destination="${repo_root}/external/gaussian_prediction"
commit="$(tr -d '[:space:]' < "${repo_root}/patches/GAUSSIANPREDICTION_COMMIT")"

if [[ ! -d "${destination}/.git" ]]; then
    mkdir -p "${repo_root}/external"
    git clone --recursive https://github.com/BoMingZhao/GaussianPrediction.git "${destination}"
fi

git -C "${destination}" fetch origin "${commit}"
git -C "${destination}" checkout --detach "${commit}"
git -C "${destination}" submodule update --init --recursive

if git -C "${destination}" apply --reverse --check \
    "${repo_root}/patches/gaussianprediction-moregs.patch" >/dev/null 2>&1; then
    echo "MoRE-GS compatibility patch is already applied."
else
    git -C "${destination}" apply --check \
        "${repo_root}/patches/gaussianprediction-moregs.patch"
    git -C "${destination}" apply \
        "${repo_root}/patches/gaussianprediction-moregs.patch"
fi

echo "GaussianPrediction ${commit} is ready at ${destination}."
echo "Run its scripts/utils/env.sh inside the intended CUDA environment."

