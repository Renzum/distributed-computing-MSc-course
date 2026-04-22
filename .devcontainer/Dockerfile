FROM ubuntu:24.04

RUN apt-get update && apt install sudo gcc g++ make cmake gdb valgrind sudo git vim bash-completion -y

# For some reason CMAKE doesn't download MPI automatically? Ask prof
RUN apt-get install -y libopenmpi-dev

RUN echo "ubuntu ALL=(ALL:ALL) NOPASSWD: ALL" | sudo tee /etc/sudoers.d/ubuntu

USER 1000

RUN echo "source /etc/profile.d/bash_completion.sh" >> /home/ubuntu/.bashrc
