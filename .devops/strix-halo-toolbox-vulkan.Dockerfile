# Strix Halo (gfx1151) toolbox: this repo built with Vulkan/RADV — the fork's
# validated performance backend. Build context = repo root, so the image always
# matches the checked-out branch. Layout follows kyuz0/amd-strix-halo-toolboxes.
#
# EXACT-ENVIRONMENT CONTRACT: this image replicates the validated on-box build
# environment (the `llama-nudge-vulkan` toolbox all published numbers came from):
# Fedora 43, gcc 15.3.1, cmake 3.31.11, glslc 2026.1, Mesa/RADV 25.3.6-3,
# vulkan-loader 1.4.341.0, and the same cmake flags as the box's build-v2 —
# including znver5 CPU codegen (the box builds GGML_NATIVE=ON on Strix Halo;
# CI runners are not znver5, so the -march is pinned explicitly instead).
# The version pins FAIL LOUDLY if Fedora updates move past them — bump them
# only together with a re-validation on the box.

# ---------- build stage ----------
FROM registry.fedoraproject.org/fedora:43 AS builder

RUN dnf -y --nodocs --setopt=install_weak_deps=False install \
  git-core patch make ninja-build libcurl-devel nodejs-npm \
  gcc-15.3.1-1.fc43 gcc-c++-15.3.1-1.fc43 cmake-3.31.11-1.fc43 \
  glslc-2026.1-1.fc43 \
  vulkan-headers-1.4.341.0-1.fc43 spirv-headers-devel-1.5.5-37.fc43 \
  vulkan-loader-1.4.341.0-1.fc43 vulkan-loader-devel-1.4.341.0-1.fc43 \
  mesa-vulkan-drivers-25.3.6-3.fc43 vulkaninfo \
  && dnf clean all && rm -rf /var/cache/dnf/*

WORKDIR /opt/llama.cpp
COPY . .

# Mirrors the box's build-v2 CMakeCache exactly (GGML_NATIVE=ON there resolves
# to -march=znver5 + the full AVX512 kernel set on Strix Halo).
RUN cmake -S . -B build -G Ninja \
  -DCMAKE_C_COMPILER=gcc -DCMAKE_CXX_COMPILER=g++ \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_INSTALL_PREFIX=/usr/local \
  -DGGML_VULKAN=ON \
  -DGGML_NATIVE=OFF \
  -DCMAKE_C_FLAGS="-march=znver5" -DCMAKE_CXX_FLAGS="-march=znver5" \
  -DGGML_AVX=ON -DGGML_AVX2=ON -DGGML_FMA=ON -DGGML_F16C=ON -DGGML_BMI2=ON \
  -DGGML_AVX512=ON -DGGML_AVX512_VBMI=ON -DGGML_AVX512_VNNI=ON -DGGML_AVX512_BF16=ON \
  -DLLAMA_USE_PREBUILT_UI=OFF \
  -DLLAMA_BUILD_TESTS=ON \
  && cmake --build build --config Release -- -j$(nproc) \
  && cmake --install build --config Release

RUN mkdir -p /usr/local/lib64 \
  && find /opt/llama.cpp/build -type f -name 'lib*.so*' -exec cp {} /usr/local/lib64/ \; \
  && ldconfig

# ---------- runtime stage ----------
FROM registry.fedoraproject.org/fedora-minimal:43

LABEL com.github.containers.toolbox="true" \
      org.opencontainers.image.source="https://github.com/wdenejko/llama.cpp" \
      org.opencontainers.image.description="llama.cpp Flash-Next fork on Vulkan/RADV for AMD Strix Halo / gfx1151 (Mesa 25.3.6, znver5)" \
      summary="Strix Halo Flash-Next toolbox (Vulkan/RADV)"

RUN microdnf -y --nodocs --setopt=install_weak_deps=0 install \
  bash ca-certificates libatomic libstdc++ libgcc libgomp sudo \
  vulkan-loader-1.4.341.0-1.fc43 vulkaninfo \
  mesa-vulkan-drivers-25.3.6-3.fc43 \
  radeontop procps-ng curl \
  && microdnf clean all && rm -rf /var/cache/dnf/*

COPY --from=builder /usr/local/ /usr/local/
COPY --from=builder /opt/llama.cpp/scripts/run/run-flashnext.sh /usr/local/share/flashnext/run-flashnext.sh

RUN echo "/usr/local/lib" > /etc/ld.so.conf.d/local.conf \
  && echo "/usr/local/lib64" >> /etc/ld.so.conf.d/local.conf \
  && ldconfig

CMD ["/bin/bash"]
