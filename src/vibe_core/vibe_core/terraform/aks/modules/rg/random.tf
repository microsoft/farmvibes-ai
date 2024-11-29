resource "random_string" "name_suffix" {
  length  = 5
  special = false
  upper   = false
  number  = false
}