
variable "m1projectid" {
    type = string
    default = "asdfasdf"
}

module "m2" {
    source = "../m2"
    m2versionyear = "2012" 
    m2versionmonth = "10" 
    m2versionday = "17" 
    m2bucketname = module.m3.fullbucketname
}
module "m3" {
    source = "../m3"
    m3bucketname = var.m1projectid
    m3environment = "dev"
}


resource "aws_s3_bucket" "bucket" {
  bucket = module.m3.fullbucketname
  policy = module.m2.fullbucketpolicy
bc-fix-0cf6e5b4-cc0c-4a7a-b0b5-d7cf44dc3fc0
   versioning {
     enabled = true
=======
   server_side_encryption_configuration {
     rule {
       apply_server_side_encryption_by_default {
         sse_algorithm = "AES256"
       }
     }
Main-renovate/k8s.io-klog-v2-2.x
   }
}
