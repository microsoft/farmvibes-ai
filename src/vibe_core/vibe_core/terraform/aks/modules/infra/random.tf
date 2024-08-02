# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

resource "random_string" "name_suffix" {
  length  = 5
  special = false
  upper   = false
  number  = false
}