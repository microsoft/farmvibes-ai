# data_ingestion/user_data/ingest_smb

```yaml

name: ingest_smb
sources:
  user_input:
  - download.user_input
sinks:
  rasters: download.rasters
parameters:
  server_name: null
  server_ip: null
  server_port: 445
  username: null
  password: null
  share_name: null
  directory_path: /
  bands:
  - red
  - green
  - blue
tasks:
  download:
    op: download_rasters_from_smb
    op_dir: download_from_smb
    parameters:
      server_name: '@from(server_name)'
      server_ip: '@from(server_ip)'
      server_port: '@from(server_port)'
      username: '@from(username)'
      password: '@from(password)'
      share_name: '@from(share_name)'
      directory_path: '@from(directory_path)'
      bands: '@from(bands)'
edges: null
description:
  short_description: Adds user rasters into the cluster storage from an SMB share,
    allowing for them to be used on workflows.
  long_description: The workflow downloads rasters from the provided SMB share and
    generates Raster objects with local assets that can be used in other operations.
  sources:
    user_input: DataVibe containing the time range and geometry metadata of the set
      rasters to be downloaded.
  sinks:
    rasters: Rasters with downloaded assets.


```

```{mermaid}
    graph TD
    inp1>user_input]
    out1>rasters]
    tsk1{{download}}
    inp1>user_input] -- user_input --> tsk1{{download}}
    tsk1{{download}} -- rasters --> out1>rasters]
```