# Prebuilts on Cerebrium

## Introduction
This repo consists of the prebuilt models listed on Cerebrium that are implemented by the Cerebrium team as well as by the Cerebrium community.

Each folder contains a Cortex deployment that can be used on Cerebrium. 

This consists of:
 - **config.yaml**: The configuration file for the cortex deployment. Contains information on the resources required for the deployment. 
 - **main.py**: The main file that is executed during the runtime
   - This has three parts
     - The `Item` which is the parameters for the API call
     - The `init` part which is only executed during the build time
     - The `predict` part which is executed during the runtime.
 - **requirements.txt**: The dependencies for the model
 - **pkglist.txt**: The list of apt packages that are installed during the build time

Additionally, it is advised to place a small README.md to describe the model and how the community can go about using it.

## How to use
### Deploy using the cerebrium dashboard. 
The primary method to deploy prebuilt models is using the dashboard.  
Simply navigate to the pre-built models tab and click `deploy` for the model you would like to deploy!

### Deploy with the Cerebrium CLI
Alternatively, you can use the Cerebrium CLI to deploy the model since each of the models in this repo are, by default, valid Cortex projects. This also allows you to customise it yourself locally.

- Clone this repo and open your terminal.  
- `cd` into the folder of the model you would like to deploy.
  
- Deploy your model with the following command:

```bash
cerebrium deploy <YOUR NAME FOR YOUR MODEL> 
```

## How to contribute
To contribute to this repo, please fork this repo and create a pull request.  
Additionally, please create an issue for any bugs or feature requests.

Don't hesitate to reach out to the team on slack or discord if there is anything you need help with or if you have any suggestions.
