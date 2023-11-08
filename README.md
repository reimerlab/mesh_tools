# mesh_tools

Utility functions for mesh representation objects (like trimesh) and manipulations using 3rd party packages (CGAL, Meshlab, etc.). All system requirements for 3rd party packages and python wrappers are performed in Dockerfile

## Setup: Installation inside docker env

---

### Download Docker Image

```bash
docker pull celiib/mesh_tools:v3
```

### Run Docker Container (from CLI)

```bash
mkdir notebooks

docker container run -it \
    -p 8890:8888 \
    -v ./notebooks:/notebooks \
    celiib/mesh_tools:v3
```

### Inside Docker Container Install Package

go to http://localhost:8890/lab and open terminal to run the folloswing commands

```bash
# install from pypi release
pip3 install mesh-processing-tools
```

OR

```bash
# install from latest github development
cd /
git clone https://github.com/reimerlab/mesh_tools
pip3 install -e /mesh_tools
```
