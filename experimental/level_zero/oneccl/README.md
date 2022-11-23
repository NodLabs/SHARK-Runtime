# Dpendecies

## oneAPI

```bash
# download the key to system keyring
wget -O- https://apt.repos.intel.com/intel-gpg-keys/GPG-PUB-KEY-INTEL-SW-PRODUCTS.PUB | gpg --dearmor | sudo tee /usr/share/keyrings/oneapi-archive-keyring.gpg > /dev/null

# add signed entry to apt sources and configure the APT client to use Intel repository:
echo "deb [signed-by=/usr/share/keyrings/oneapi-archive-keyring.gpg] https://apt.repos.intel.com/oneapi all main" | sudo tee /etc/apt/sources.list.d/oneAPI.list
sudo apt-get update
sudo apt-get install intel-basekit intel-oneapi-ccl-devel libnuma-dev
```

Before building or running programs you would need to
```bash
source /opt/intel/oneapi/setvars.sh
```

# CMake Configuration

To enable to oneCCL as a collectives backend for the Level Zero driver
you would need to pass to cmake
```
-DIREE_BUILD_EXPERIMENTAL_HAL_DRIVER_LEVEL_ZERO_ONECCL=ON
-DoneCCL_DIR=/opt/intel/oneapi/ccl/latest
-DSYCL_C_COMPILER=/opt/intel/oneapi/compiler/latest/linux/bin/icx
-DSYCL_CXX_COMPILER=/opt/intel/oneapi/compiler/latest/linux/bin/dpcpp
``
