
FROM almalinux:9 as base

SHELL ["/bin/bash", "-c"]


ENV TZ=Etc/UTC
# Copy the environment files
COPY requirements.txt /docker/requirements.txt
COPY environment.yml /docker/environment.yml 
COPY full.conda-lock.yml /docker/conda-lock.yml 
COPY requirements.lock /docker/requirements.lock 

# Installing system dependencies and creation necessary directories
RUN yum install -y bzip2 && \
    yum clean all && \
    yum autoremove -y && \
    curl -Ls https://micro.mamba.pm/api/micromamba/linux-64/latest | tar -xvj -C /usr/local/bin/ --strip-components=1 bin/micromamba && \
    micromamba shell init --prefix /opt/micromamba --shell bash && \
    /usr/local/bin/micromamba shell init --prefix /opt/micromamba && \
    cp /root/.bashrc /etc/.bashrc && \
    . /etc/.bashrc && \
    micromamba create --name lock && \
    micromamba activate lock && \
    micromamba env list && \
    micromamba install --channel conda-forge --yes conda-lock && \
    conda-lock install --help && \
    conda-lock install --micromamba --name analysis-systems /docker/conda-lock.yml && \
    micromamba clean --yes --all && \
    echo 'export PATH="/opt/micromamba/bin:${PATH}"' >> /etc/.bashrc && \
    echo "micromamba activate analysis-systems" >> /etc/.bashrc && \
    echo 'if [ -f /etc/.bashrc ]; then . /etc/.bashrc; fi' >> /etc/profile && \
    . /etc/.bashrc && \
    micromamba activate analysis-systems && \
    rm -rf /opt/micromamba/envs/lock && \
    chmod --recursive 777 /opt/micromamba/envs && \
    micromamba env list && \
    micromamba install --channel conda-forge --freeze-installed --yes cmake gcc git && \
    micromamba clean --yes --all && \
    python -m pip --no-cache-dir install --upgrade pip setuptools wheel && \
    python -m pip --no-cache-dir install --no-deps --require-hashes --only-binary :all: --no-binary torch-geometric --use-pep517 --requirement /docker/requirements.lock && \
    python -m pip --no-cache-dir install --find-links https://data.pyg.org/whl/torch-1.12.1+cpu.html 'torch-scatter==2.1.0+pt112cpu' 'torch-sparse==0.6.15+pt112cpu' 'torch-cluster==1.6.0+pt112cpu' 'torch-spline-conv==1.2.1+pt112cpu' && \
    python -m ipykernel install --name="analysis-systems" --display-name="Analysis Systems" --sys-prefix && \
    rm -rf /root/*

# Creating necessary directories and symbolic links
RUN mkdir -p -v /etc/condor && \
    mkdir -p -v /direct/condor && \
    mkdir -p -v /usatlas/atlas01 && \
    ln -sf /opt/micromamba/envs/analysis-systems/etc/grid-security /etc/grid-security && \
    ln -sf /lib64/libcrypt.so.2 /lib64/libcrypt.so.1 && \
    ln -sf /lib64/libcrypto.so.3 /lib64/libcrypto.so.10 && \
    ln -sf /lib64/libssl.so.3 /lib64/libssl.so.10 && \
    mkdir -p -v /u0b/software/jupyter/kernels && \
    ln -s /opt/micromamba/envs/analysis-systems/share/jupyter/kernels/analysis-systems /u0b/software/jupyter/kernels/analysis-systems && \
    mkdir -p -v /.local /.jupyter /.config /.cache /work && \
    chmod --recursive 777 /.local /.jupyter /.config /.cache /work && \
    printf '#!/bin/bash\n\njupyter lab --no-browser --ip 0.0.0.0 --port 8888\n' > /docker/entrypoint.sh && \
    chmod 777 /docker/entrypoint.sh && \
    echo 'export JUPYTER_PATH="/opt/micromamba/envs/analysis-systems/share/jupyter:${JUPYTER_PATH}"' >> /etc/.bashrc && \
    echo 'export JUPYTER_DATA_DIR="/opt/micromamba/envs/analysis-systems/share/jupyter"' >> /etc/.bashrc && \
    chmod --recursive 777 /opt/micromamba/envs/analysis-systems/share/jupyter && \
    echo "# image build date: $(date)" > /docker/build_date.txt

WORKDIR /work

ENV HOME=/work

SHELL ["/bin/bash", "-lc"]

# Installing papermill for 
RUN micromamba install papermill --channel conda-forge 


COPY entrypoint.sh /entrypoint.sh  
RUN chmod +x /entrypoint.sh

ENTRYPOINT ["/entrypoint.sh"]

CMD ["/docker/entrypoint.sh"]
