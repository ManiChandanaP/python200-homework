# Q1
#
# When running locally, DefaultAzureCredential typically uses the Azure CLI
# credential if I have already authenticated with Azure using:
#
#     az login
#
# The Azure CLI stores authentication tokens on the local machine.
# DefaultAzureCredential checks multiple authentication sources in a specific
# order and automatically discovers the Azure CLI credential when it is
# available.

# Q2
#
# A deployed application cannot rely on az login because there is no human
# available to perform an interactive login. Instead, Azure resources usually
# use a Managed Identity.
#
# DefaultAzureCredential automatically detects the environment and uses the
# Managed Identity when running in Azure. This allows the exact same Python
# code to work locally and in production without modification.


# Q3
#
# Two common causes of AuthenticationError are:
#
# 1. The user has not authenticated with Azure CLI.
#    Diagnosis:
#       Run:
#           az account show
#       If this fails, run:
#           az login
#
# 2. The authenticated identity does not have permission to access the
#    requested resource.
#    Diagnosis:
#       Verify RBAC permissions in Azure Portal or inspect role assignments
#       using Azure CLI commands.

# --- Blob Storage ---
# Q1
#
# Azure Blob Storage has a three-level hierarchy:
#
# 1. Storage Account
# 2. Container
# 3. Blob
#
# A useful analogy is a filing cabinet:
#
# Storage Account = Filing Cabinet
# Container = Drawer
# Blob = Individual Document
#
# The storage account contains containers, and each container contains blobs.

# Q2
#
# REST API JSON responses:
# Use Blob Storage because raw files are inexpensive to store and easy to
# reprocess later.
#
# 50 million customer transactions:
# Use a relational database because the data will be queried frequently by
# customer ID and date range.
#
# NumPy embeddings:
# Use Blob Storage because embeddings are large binary objects that do not
# require relational queries.

#Q3
def list_container(container_client):
    """
    Print every blob's name and size.
    """

    for blob in container_client.list_blobs():
        print(f"{blob.name}: {blob.size} bytes")

#Q4
def upload_text(container_client, blob_name, text):
    """
    Upload UTF-8 text as a blob.
    """

    data = text.encode("utf-8")

    container_client.upload_blob(
        name=blob_name,
        data=data,
        overwrite=True
    )