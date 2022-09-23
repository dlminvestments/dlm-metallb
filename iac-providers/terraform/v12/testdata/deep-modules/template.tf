terraform {
  required_version = ">= 0.12.0"
}

provider "aws" {
renovate/aws-4.x
  version = "4.3.0"
=======
  version = "4.32.0"
Main-renovate/k8s.io-klog-v2-2.x
  region  = "us-east-1"
}


module "m1" {
    source = "./modules/m1"
    m1projectid = "tf-test-project"
}
