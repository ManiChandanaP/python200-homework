

For Scenario A, I estimated the cost of a lightweight Linux VM using the Standard_B1s instance running about 160 hours per month. The total monthly cost was very low, only a few dollars, which shows how affordable small cloud workloads can be when resources are limited and not running continuously.

For Scenario B, I estimated a much heavier analytics workload using a Standard_NC6s_v3 GPU-enabled VM running 24/7, along with an Azure SQL Database and 1 TB of Blob Storage. The GPU virtual machine alone cost over $2200 per month because it includes a powerful NVIDIA V100 GPU designed for machine learning and high-performance computing workloads. Adding the SQL Database and Blob Storage increased the total monthly estimate even further.

One interesting thing I noticed while exploring the Azure Pricing Calculator was how dramatically costs increase when using GPUs, larger databases, or always-on infrastructure. Small changes in VM size or runtime hours can significantly affect the monthly cost. It made me realize how important budgeting and cost planning are in cloud environments.

The Python script successfully calculated the VM costs using the hourly rates from the Pricing Calculator. The calculated monthly totals closely matched the estimates shown in Azure.

video link - https://youtu.be/ZhvaL4EerNI 