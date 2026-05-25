Cloud Concepts Question 1
------------------------
The core economic model of cloud computing is pay-as-you-go, where you only pay for the resources you use.
This is different from owning servers because with your own servers you must buy, maintain, and upgrade the hardware yourself.

Cloud Concepts Question 2
-------------------------
Vertical scaling means making one machine more powerful by adding more CPU, RAM, or storage.
Horizontal scaling means adding more machines to share the workload.

Better to choose the vertical scaling for a database that needs more memory, and horizontal scaling for a website with many users.

Scenarios
The web app uses horizontal scaling because more servers can be added to handle many new users.
The data scientist uses vertical scaling because they need one stronger machine with better hardware.
The data pipeline uses horizontal scaling because the file processing can be split across many machines.

Cloud Concepts Question 3
--------------------------
Classification
Gmail — SaaS because users simply use the software through the internet.
Microsoft Azure Virtual Machines — IaaS because you manage the operating system and software yourself.
Microsoft Azure App Service — PaaS because Azure manages the infrastructure while you deploy the app.
Amazon Web Services S3 — IaaS because it provides cloud storage infrastructure.
GitHub Codespaces — PaaS because it gives developers a managed coding environment.
Snowflake — SaaS because it is fully managed software for data analytics.
Definitions
IaaS gives you cloud infrastructure like servers and storage, but you manage the software and operating system.
Example: Azure Virtual Machines — you manage the OS, apps, and updates.
PaaS gives you a platform to build and deploy apps without managing servers.
Example: Azure App Service — you manage your app code while Azure manages the servers.
SaaS gives you ready-to-use software over the internet.
Example: Gmail — you only use the software while Google manages everything else.

Cloud Concepts Question 4
------------------------
A managed data platform like Databricks or Snowflake handles infrastructure, scaling, and maintenance for you.
Compared to using Azure directly, you gain simplicity and faster setup, but you give up some control and flexibility.

Cloud Concepts Question 5
-------------------------
The cloud is probably not the right choice when you need complete control over hardware or when strict legal/security rules require data to stay on-site.

Azure Basics
Azure Basics Question 1
-----------------------
An Azure subscription is the billing and account container for all resources, while a resource group is a smaller container for related resources.

Azure Basics Question 2
-----------------------
Ephemeral means Cloud Shell storage disappears when the session ends unless it is saved somewhere persistent.
The course setup uses attached cloud storage to keep files and settings saved.

Azure Basics Question 3
-----------------------
The SSH private key is secret and stays on your computer, while the SSH public key can be shared safely.
The public key gets uploaded to remote systems because it can only verify identity and cannot be used to access your computer.

Azure Basics Question 4

Example output:

{
  "environmentName": "AzureCloud",
  "homeTenantId": "xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx",
  "id": "xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx",
  "isDefault": true,
  "managedByTenants": [],
  "name": "My Subscription",
  "state": "Enabled",
  "tenantId": "xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx",
  "user": {
    "name": "example@example.com",
    "type": "user"
  }
}
Adding --output table changes the result from JSON format to a simple table that is easier to read.