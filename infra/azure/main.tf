# Azure deployment: the same container on Azure Container Apps, calling Azure OpenAI
# with a managed identity (no API keys in config). Postgres Flexible Server with pgvector.

terraform {
  required_version = ">= 1.6"
  required_providers {
    azurerm = { source = "hashicorp/azurerm", version = "~> 3.100" }
  }
}

provider "azurerm" {
  features {}
}

variable "location" { default = "eastus2" }
variable "env" { default = "staging" }
variable "image" {
  type        = string
  description = "Full image reference, e.g. <registry>.azurecr.io/bluewhale:<tag>"
}
variable "openai_account_id" { type = string }
variable "openai_endpoint" { type = string }
variable "openai_deployment" { default = "gpt-4o-mini" }
variable "key_vault_id" { type = string }
variable "db_url_secret_id" {
  type        = string
  description = "Key Vault secret id holding BW_DATABASE_URL"
}

locals {
  name = "bluewhale-${var.env}"
}

resource "azurerm_resource_group" "this" {
  name     = "rg-${local.name}"
  location = var.location
}

resource "azurerm_log_analytics_workspace" "this" {
  name                = "log-${local.name}"
  location            = azurerm_resource_group.this.location
  resource_group_name = azurerm_resource_group.this.name
  sku                 = "PerGB2018"
  retention_in_days   = 30
}

resource "azurerm_user_assigned_identity" "api" {
  name                = "id-${local.name}"
  location            = azurerm_resource_group.this.location
  resource_group_name = azurerm_resource_group.this.name
}

# Least privilege: the app can call the model deployments, nothing else on the account.
resource "azurerm_role_assignment" "openai_user" {
  scope                = var.openai_account_id
  role_definition_name = "Cognitive Services OpenAI User"
  principal_id         = azurerm_user_assigned_identity.api.principal_id
}

resource "azurerm_role_assignment" "kv_reader" {
  scope                = var.key_vault_id
  role_definition_name = "Key Vault Secrets User"
  principal_id         = azurerm_user_assigned_identity.api.principal_id
}

resource "azurerm_container_app_environment" "this" {
  name                       = "cae-${local.name}"
  location                   = azurerm_resource_group.this.location
  resource_group_name        = azurerm_resource_group.this.name
  log_analytics_workspace_id = azurerm_log_analytics_workspace.this.id
}

resource "azurerm_container_app" "api" {
  name                         = "ca-${local.name}"
  container_app_environment_id = azurerm_container_app_environment.this.id
  resource_group_name          = azurerm_resource_group.this.name
  revision_mode                = "Single"

  identity {
    type         = "UserAssigned"
    identity_ids = [azurerm_user_assigned_identity.api.id]
  }

  secret {
    name                = "database-url"
    key_vault_secret_id = var.db_url_secret_id
    identity            = azurerm_user_assigned_identity.api.id
  }

  template {
    min_replicas = 1
    max_replicas = 5

    container {
      name   = "api"
      image  = var.image
      cpu    = 1.0
      memory = "2Gi"

      env {
        name  = "BW_ENV"
        value = var.env
      }
      env {
        name  = "BW_LLM_PROVIDERS"
        value = "azure_openai,bedrock"
      }
      env {
        name  = "AZURE_OPENAI_ENDPOINT"
        value = var.openai_endpoint
      }
      env {
        name  = "AZURE_OPENAI_DEPLOYMENT"
        value = var.openai_deployment
      }
      env {
        name  = "AZURE_CLIENT_ID"
        value = azurerm_user_assigned_identity.api.client_id
      }
      env {
        name  = "BW_VECTOR_BACKEND"
        value = "pgvector"
      }
      env {
        name        = "BW_DATABASE_URL"
        secret_name = "database-url"
      }

      readiness_probe {
        transport = "HTTP"
        port      = 8000
        path      = "/readyz"
      }
      liveness_probe {
        transport = "HTTP"
        port      = 8000
        path      = "/healthz"
      }
    }
  }

  ingress {
    external_enabled = true
    target_port      = 8000
    traffic_weight {
      latest_revision = true
      percentage      = 100
    }
  }
}

output "api_url" {
  value = "https://${azurerm_container_app.api.ingress[0].fqdn}"
}
