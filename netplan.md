sudo apt-get install netplan.io

# Let NetworkManager manage all devices on this system
network:
  version: 2
  renderer: NetworkManager
  ethernets:
    eth0:
      addresses: [192.168.2.194/24]
      addresses: [192.168.0.194/24]
      gateway4: 192.168.2.15