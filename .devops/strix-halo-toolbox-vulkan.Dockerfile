# Strix Halo (gfx1151) toolbox: this repo built with Vulkan/RADV — the fork's
# validated performance backend. Build context = repo root, so the image always
# matches the checked-out branch. Layout follows kyuz0/amd-strix-halo-toolboxes.

# ---------- build stage ----------
FROM registry.fedoraproject.org/fedora:43 AS builder

RUN dnf -y --nodocs --setopt=install_weak_deps=False install \
  git-core patch \
  make gcc gcc-c++ cmake ninja-build lld clang compiler-rt libcurl-devel \
  vulkan-loader-devel vulkaninfo mesa-vulkan-drivers \
  spirv-headers-devel glslc \
  && dnf clean all && rm -rf /var/cache/dnf/*

WORKDIR /opt/llama.cpp
COPY . .

RUN cmake -S . -B build -G Ninja \
  -DGGML_VULKAN=ON \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_INSTALL_PREFIX=/usr/local \
  -DLLAMA_USE_PREBUILT_UI=OFF \
  -DLLAMA_BUILD_TESTS=OFF \
  && cmake --build build --config Release -- -j$(nproc) \
  && cmake --install build --config Release

RUN mkdir -p /usr/local/lib64 \
  && find /opt/llama.cpp/build -type f -name 'lib*.so*' -exec cp {} /usr/local/lib64/ \; \
  && ldconfig

# ---------- runtime stage ----------
FROM registry.fedoraproject.org/fedora-minimal:43

LABEL com.github.containers.toolbox="true" \
      org.opencontainers.image.source="https://github.com/wdenejko/llama.cpp" \
      org.opencontainers.image.description="llama.cpp Flash-Next fork on Vulkan/RADV for AMD Strix Halo / gfx1151" \
      summary="Strix Halo Flash-Next toolbox (Vulkan/RADV)"

RUN microdnf -y --nodocs --setopt=install_weak_deps=0 install \
  bash ca-certificates libatomic libstdc++ libgcc sudo \
  vulkan-loader vulkaninfo mesa-vulkan-drivers \
  radeontop procps-ng curl \
  && microdnf clean all && rm -rf /var/cache/dnf/*

COPY --from=builder /usr/local/ /usr/local/
COPY --from=builder /opt/llama.cpp/scripts/run/run-flashnext.sh /usr/local/share/flashnext/run-flashnext.sh

RUN echo "/usr/local/lib" > /etc/ld.so.conf.d/local.conf \
  && echo "/usr/local/lib64" >> /etc/ld.so.conf.d/local.conf \
  && ldconfig

CMD ["/bin/bash"]
