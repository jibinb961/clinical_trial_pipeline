Google Cloud Run Worker Guide
​
Why use Google Cloud Run for flow run execution?

Google Cloud Run is a fully managed compute platform that automatically scales your containerized applications.

Serverless architecture: Cloud Run follows a serverless architecture, which means you don’t need to manage any underlying infrastructure. Google Cloud Run automatically handles the scaling and availability of your flow run infrastructure, allowing you to focus on developing and deploying your code.

Scalability: Cloud Run can automatically scale your pipeline to handle varying workloads and traffic. It can quickly respond to increased demand and scale back down during low activity periods, ensuring efficient resource utilization.

Integration with Google Cloud services: Google Cloud Run easily integrates with other Google Cloud services, such as Google Cloud Storage, Google Cloud Pub/Sub, and Google Cloud Build. This interoperability enables you to build end-to-end data pipelines that use a variety of services.

Portability: Since Cloud Run uses container images, you can develop your pipelines locally using Docker and then deploy them on Google Cloud Run without significant modifications. This portability allows you to run the same pipeline in different environments.

​
Google Cloud Run guide

After completing this guide, you will have:

Created a Google Cloud Service Account
Created a Prefect Work Pool
Deployed a Prefect Worker as a Cloud Run Service
Deployed a Flow
Executed the Flow as a Google Cloud Run Job
​
Prerequisites

Before starting this guide, make sure you have:

A Google Cloud Platform (GCP) account.
A project on your GCP account where you have the necessary permissions to create Cloud Run Services and Service Accounts.
The gcloud CLI installed on your local machine. You can follow Google Cloud’s installation guide. If you’re using Apple (or a Linux system) you can also use Homebrew for installation.
Docker installed on your local machine.
A Prefect server instance. You can sign up for a forever free Prefect Cloud Account or, alternatively, self-host a Prefect server.
​
Step 1. Create a Google Cloud service account

First, open a terminal or command prompt on your local machine where gcloud is installed. If you haven’t already authenticated with gcloud, run the following command and follow the instructions to log in to your GCP account.


Copy
gcloud auth login
Next, you’ll set your project where you’d like to create the service account. Use the following command and replace <PROJECT_ID> with your GCP project’s ID.


Copy
gcloud config set project <PROJECT-ID>
For example, if your project’s ID is prefect-project the command will look like this:


Copy
gcloud config set project prefect-project
Now you’re ready to make the service account. To do so, you’ll need to run this command:


Copy
gcloud iam service-accounts create <SERVICE-ACCOUNT-NAME> --display-name="<DISPLAY-NAME>"
Here’s an example of the command above which you can use which already has the service account name and display name provided. An additional option to describe the service account has also been added:


Copy
gcloud iam service-accounts create prefect-service-account \
    --description="service account to use for the prefect worker" \
    --display-name="prefect-service-account"
The last step of this process is to make sure the service account has the proper permissions to execute flow runs as Cloud Run jobs. Run the following commands to grant the necessary permissions:


Copy
gcloud projects add-iam-policy-binding <PROJECT-ID> \
    --member="serviceAccount:<SERVICE-ACCOUNT-NAME>@<PROJECT-ID>.iam.gserviceaccount.com" \
    --role="roles/iam.serviceAccountUser"

Copy
gcloud projects add-iam-policy-binding <PROJECT-ID> \
    --member="serviceAccount:<SERVICE-ACCOUNT-NAME>@<PROJECT-ID>.iam.gserviceaccount.com" \
    --role="roles/run.admin"
​
Step 2. Create a Cloud Run work pool

Let’s walk through the process of creating a Cloud Run work pool.

​
Fill out the work pool base job template

You can create a new work pool using the Prefect UI or CLI. The following command creates a work pool of type cloud-run via the CLI (you’ll want to replace the <WORK-POOL-NAME> with the name of your work pool):


Copy
prefect work-pool create --type cloud-run <WORK-POOL-NAME>
Once the work pool is created, find the work pool in the UI and edit it.

There are many ways to customize the base job template for the work pool. Modifying the template influences the infrastructure configuration that the worker provisions for flow runs submitted to the work pool. For this guide we are going to modify just a few of the available fields.

Specify the region for the cloud run job.

region

Save the name of the service account created in first step of this guide.

name

Your work pool is now ready to receive scheduled flow runs!

​
Step 3. Deploy a Cloud Run worker

Now you can launch a Cloud Run service to host the Cloud Run worker. This worker will poll the work pool that you created in the previous step.

Navigate back to your terminal and run the following commands to set your Prefect API key and URL as environment variables. Be sure to replace <ACCOUNT-ID> and <WORKSPACE-ID> with your Prefect account and workspace IDs (both will be available in the URL of the UI when previewing the workspace dashboard). You’ll want to replace <YOUR-API-KEY> with an active API key as well.


Copy
export PREFECT_API_URL='https://api.prefect.cloud/api/accounts/<ACCOUNT-ID>/workspaces/<WORKSPACE-ID>'
export PREFECT_API_KEY='<YOUR-API-KEY>'
Once those variables are set, run the following shell command to deploy your worker as a service. Don’t forget to replace <YOUR-SERVICE-ACCOUNT-NAME> with the name of the service account you created in the first step of this guide, and replace <WORK-POOL-NAME> with the name of the work pool you created in the second step.


Copy
gcloud run deploy prefect-worker --image=prefecthq/prefect:3-latest \
--set-env-vars PREFECT_API_URL=$PREFECT_API_URL,PREFECT_API_KEY=$PREFECT_API_KEY \
--service-account <YOUR-SERVICE-ACCOUNT-NAME> \
--no-cpu-throttling \
--min-instances 1 \
--startup-probe httpGet.port=8080,httpGet.path=/health,initialDelaySeconds=100,periodSeconds=20,timeoutSeconds=20 \
--args "prefect","worker","start","--install-policy","always","--with-healthcheck","-p","<WORK-POOL-NAME>","-t","cloud-run"
After running this command, you’ll be prompted to specify a region. Choose the same region that you selected when creating the Cloud Run work pool in the second step of this guide. The next prompt will ask if you’d like to allow unauthenticated invocations to your worker. For this guide, you can select “No”.

After a few seconds, you’ll be able to see your new prefect-worker service by navigating to the Cloud Run page of your Google Cloud console. Additionally, you should be able to see a record of this worker in the Prefect UI on the work pool’s page by navigating to the Worker tab. Let’s not leave our worker hanging, it’s time to give it a job.

​
Step 4. Deploy a flow

Let’s prepare a flow to run as a Cloud Run job. In this section of the guide, we’ll “bake” our code into a Docker image, and push that image to Google Artifact Registry.

​
Create a registry

Let’s create a docker repository in your Google Artifact Registry to host your custom image. If you already have a registry, and are authenticated to it, skip ahead to the Write a flow section.

The following command creates a repository using the gcloud CLI. You’ll want to replace the <REPOSITORY-NAME> with your own value. :


Copy
gcloud artifacts repositories create <REPOSITORY-NAME> \
--repository-format=docker --location=us
Now you can authenticate to artifact registry:


Copy
gcloud auth configure-docker us-docker.pkg.dev
​
Write a flow

First, create a new directory. This will serve as the root of your project’s repository. Within the directory, create a sub-directory called flows. Navigate to the flows subdirectory and create a new file for your flow. Feel free to write your own flow, but here’s a ready-made one for your convenience:


Copy
import httpx
from prefect import flow, task
from prefect.artifacts import create_markdown_artifact

@task
def mark_it_down(temp):
    markdown_report = f"""# Weather Report
## Recent weather

| Time      | Temperature |
| :-------- | ----------: |
| Now       |      {temp} |
| In 1 hour |  {temp + 2} |
"""
    create_markdown_artifact(
        key="weather-report",
        markdown=markdown_report,
        description="Very scientific weather report",
    )


@flow
def fetch_weather(lat: float, lon: float):
    base_url = "https://api.open-meteo.com/v1/forecast/"
    weather = httpx.get(
        base_url,
        params=dict(latitude=lat, longitude=lon, hourly="temperature_2m"),
    )
    most_recent_temp = float(weather.json()["hourly"]["temperature_2m"][0])
    mark_it_down(most_recent_temp)


if __name__ == "__main__":
    fetch_weather(38.9, -77.0)
In the remainder of this guide, this script will be referred to as weather_flow.py, but you can name yours whatever you’d like.

​
Creating a prefect.yaml file

Now we’re ready to make a prefect.yaml file, which will be responsible for managing the deployments of this repository. Navigate back to the root of your directory, and run the following command to create a prefect.yaml file using Prefect’s docker deployment recipe.


Copy
prefect init --recipe docker
You’ll receive a prompt to put in values for the image name and tag. Since we will be pushing the image to Google Artifact Registry, the name of your image should be prefixed with the path to the docker repository you created within the registry. For example: us-docker.pkg.dev/<PROJECT-ID>/<REPOSITORY-NAME>/. You’ll want to replace <PROJECT-ID> with the ID of your project in GCP. This should match the ID of the project you used in first step of this guide. Here is an example of what this could look like:


Copy
image_name: us-docker.pkg.dev/prefect-project/my-artifact-registry/gcp-weather-image
tag: latest
At this point, there will be a new prefect.yaml file available at the root of your project. The contents will look similar to the example below, however, we’ve added in a combination of YAML templating options and Prefect deployment actions to build out a simple CI/CD process. Feel free to copy the contents and paste them in your prefect.yaml:


Copy
# Welcome to your prefect.yaml file! You can you this file for storing and managing
# configuration for deploying your flows. We recommend committing this file to source
# control along with your flow code.

# Generic metadata about this project
name: <WORKING-DIRECTORY>
prefect-version: 3.0.0

# build section allows you to manage and build docker image
build:
- prefect_docker.deployments.steps.build_docker_image:
    id: build_image
    requires: prefect-docker>=0.3.1
    image_name: <PATH-TO-ARTIFACT-REGISTRY>/gcp-weather-image
    tag: latest
    dockerfile: auto
    platform: linux/amd64

# push section allows you to manage if and how this project is uploaded to remote locations
push:
- prefect_docker.deployments.steps.push_docker_image:
    requires: prefect-docker>=0.3.1
    image_name: '{{ build_image.image_name }}'
    tag: '{{ build_image.tag }}'

# pull section allows you to provide instructions for cloning this project in remote locations
pull:
- prefect.deployments.steps.set_working_directory:
    directory: /opt/prefect/<WORKING-DIRECTORY>

# the deployments section allows you to provide configuration for deploying flows
deployments:
- name: gcp-weather-deploy
  version: null
  tags: []
  description: null
  schedule: {}
  flow_name: null
  entrypoint: flows/weather_flow.py:fetch_weather
  parameters:
    lat: 14.5994
    lon: 28.6731
  work_pool:
    name: my-cloud-run-pool
    work_queue_name: default
    job_variables:
      image: '{{ build_image.image }}'
After copying the example above, don’t forget to replace <WORKING-DIRECTORY> with the name of the directory where your flow folder and prefect.yaml live. You’ll also need to replace <PATH-TO-ARTIFACT-REGISTRY> with the path to the Docker repository in your Google Artifact Registry.
To get a better understanding of the different components of the prefect.yaml file above and what they do, feel free to read this next section. Otherwise, you can skip ahead to Flow Deployment.

In the build section of the prefect.yaml the following step is executed at deployment build time:

prefect_docker.deployments.steps.build_docker_image : builds a Docker image automatically which uses the name and tag chosen previously.
If you are using an ARM-based chip (such as an M1 or M2 Mac), you’ll want to ensure that you add platform: linux/amd64 to your build_docker_image step to ensure that your docker image uses an AMD architecture. For example:


Copy
- prefect_docker.deployments.steps.build_docker_image:
id: build_image
requires: prefect-docker>=0.3.1
image_name: us-docker.pkg.dev/prefect-project/my-docker-repository/gcp-weather-image
tag: latest
dockerfile: auto
platform: linux/amd64
The push section sends the Docker image to the Docker repository in your Google Artifact Registry, so that it can be easily accessed by the worker for flow run execution.

The pull section sets the working directory for the process prior to importing your flow.

In the deployments section of the prefect.yaml file above, you’ll see that there is a deployment declaration named gcp-weather-deploy. Within the declaration, the entrypoint for the flow is specified along with some default parameters which will be passed to the flow at runtime. Last but not least, the name of the work pool that we created in step 2 of this guide is specified.

​
Flow deployment

Once you’re happy with the specifications in the prefect.yaml file, run the following command in the terminal to deploy your flow:


Copy
prefect deploy --name gcp-weather-deploy
Once the flow is deployed to Prefect Cloud or your local Prefect Server, it’s time to queue up a flow run!

​
Step 5. Flow execution

Find your deployment in the UI, and hit the Quick Run button. You have now successfully submitted a flow run to your Cloud Run worker! If you used the flow script provided in this guide, check the Artifacts tab for the flow run once it completes. You’ll have a nice little weather report waiting for you there. Hope your day is a sunny one!

​
Recap and next steps

Congratulations on completing this guide! Looking back on our journey, you have:

Created a Google Cloud service account
Created a Cloud Run work pool
Deployed a Cloud Run worker
Deployed a flow
Executed a flow
For next steps, take a look at some of the other work pools Prefect has to offer.

The world is your oyster 🦪✨.