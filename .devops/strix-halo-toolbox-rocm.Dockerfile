# Strix Halo (gfx1151) toolbox: this repo built with ROCm 7.14 / HIP, from AMD's
# TheRock release-stream packages. Comparison/prefill-experiment build — Vulkan
# is the fork's validated backend. Build context = repo root.

# ---------- build stage ----------
FROM registry.fedoraproject.org/fedora:44 AS builder

RUN <<'EOF'
tee /etc/yum.repos.d/rocm.repo <<REPO
[rocm]
name=ROCm 7.14.0
baseurl=https://repo.amd.com/rocm/packages-multi-arch/rhel10/x86_64
enabled=1
priority=50
gpgcheck=1
gpgkey=https://repo.amd.com/rocm/packages-multi-arch/gpg/rocm.gpg
REPO
EOF

RUN dnf -y --nodocs --setopt=install_weak_deps=False install \
  make gcc gcc-c++ cmake ninja-build libcurl-devel \
  amdrocm-core-devel7.14-gfx1151 \
  git-core patch \
  && dnf clean all && rm -rf /var/cache/dnf/*

ENV ROCM_PATH=/opt/rocm \
  HIP_PATH=/opt/rocm \
  PATH=/opt/rocm/bin:/opt/rocm/core/bin:/opt/rocm/core/lib/llvm/bin:$PATH \
  LD_LIBRARY_PATH=/opt/rocm/core/lib/rocm_sysdeps/lib:/opt/rocm/core/lib

WORKDIR /opt/llama.cpp
COPY . .

RUN cmake -S . -B build -G Ninja \
  -DGGML_HIP=ON \
  -DAMDGPU_TARGETS=gfx1151 \
  -DCMAKE_BUILD_TYPE=Release \
  -DROCM_PATH=/opt/rocm \
  -DHIP_PLATFORM=amd \
  -DCMAKE_INSTALL_PREFIX=/usr/local \
  -DLLAMA_USE_PREBUILT_UI=OFF \
  -DLLAMA_BUILD_TESTS=OFF \
  && cmake --build build --config Release -- -j$(nproc) \
  && cmake --install build --config Release

RUN mkdir -p /usr/local/lib64 \
  && find /opt/llama.cpp/build -type f -name 'lib*.so*' -exec cp {} /usr/local/lib64/ \; \
  && ldconfig

# ---------- runtime stage ----------
FROM registry.fedoraproject.org/fedora-minimal:44

LABEL com.github.containers.toolbox="true" \
      org.opencontainers.image.source="https://github.com/wdenejko/llama.cpp" \
      org.opencontainers.image.description="llama.cpp Flash-Next fork on ROCm 7.14 for AMD Strix Halo / gfx1151" \
      summary="Strix Halo Flash-Next toolbox (ROCm 7.14)"

RUN <<'EOF'
tee /etc/yum.repos.d/rocm.repo <<REPO
[rocm]
name=ROCm 7.14.0
baseurl=https://repo.amd.com/rocm/packages-multi-arch/rhel10/x86_64
enabled=1
priority=50
gpgcheck=1
gpgkey=https://repo.amd.com/rocm/packages-multi-arch/gpg/rocm.gpg
REPO
EOF

RUN microdnf -y --nodocs --setopt=install_weak_deps=0 install \
  bash ca-certificates libatomic libstdc++ libgcc libgomp sudo \
  amdrocm-runtime7.14 amdrocm-blas7.14-gfx1151 \
  radeontop procps-ng curl \
  && microdnf clean all && rm -rf /var/cache/dnf/* \
  && ln -s core-7.14 /opt/rocm/core \
  && ln -s core-7.14/bin /opt/rocm/bin \
  && ln -s core-7.14/include /opt/rocm/include \
  && ln -s core-7.14/lib /opt/rocm/lib \
  && ln -s core-7.14/libexec /opt/rocm/libexec \
  && ln -s core-7.14/lib/llvm /opt/rocm/llvm \
  && ln -s core-7.14/share /opt/rocm/share \
  && ln -s core-7.14/lib/llvm/amdgcn /opt/rocm/amdgcn

ENV ROCM_PATH=/opt/rocm \
  HIP_PATH=/opt/rocm \
  PATH=/opt/rocm/bin:/opt/rocm/core/bin:$PATH \
  LD_LIBRARY_PATH=/opt/rocm/core/lib/rocm_sysdeps/lib:/opt/rocm/core/lib

# The fork's fused HC-mix tail is a Vulkan+CPU custom op; on HIP it would fall
# back to a single-thread CPU reference per token. Keep the primitive chain.
ENV Q4X_HC_MIX_NOFUSE=1

COPY --from=builder /usr/local/ /usr/local/
COPY --from=builder /opt/llama.cpp/scripts/run/run-flashnext.sh /usr/local/share/flashnext/run-flashnext.sh

RUN echo "/usr/local/lib" > /etc/ld.so.conf.d/local.conf \
  && echo "/usr/local/lib64" >> /etc/ld.so.conf.d/local.conf \
  && echo "/opt/rocm/core/lib" > /etc/ld.so.conf.d/rocm.conf \
  && echo "/opt/rocm/core/lib/rocm_sysdeps/lib" >> /etc/ld.so.conf.d/rocm.conf \
  && ldconfig

CMD ["/bin/bash"]
