# Notes on Setting Up



## Documentation

To automatically generate the API documentation from the docstrings,

```sh
cd doc; sphinx-quickstart
cd ..; sphinx-apidoc -f -o doc mclearn
```

and the doc can be built by

```sh
cd doc; make html
```

## Development

During development, if we want the changes we make to the package to take effect immediately without reinstalling, then provide the `develop` argument when first installing the package

```sh
python setup.py develop
```

and in the Jupyter notebooks, the `autoreload` extension will automatically re-import all the packages every time a code cell is run:

```python
%load_ext autoreload
%autoreload 2
```

## Amazon EC2

Some algorithms are fairly computationally expensive, espcially on the bigger datasets. One option to speed up experiments is to rent a spot instance on Amazon EC2. To start, log into the [AWS management console](https://console.aws.amazon.com/) and click over to EC2. Then choose a region - US West (Oregon) tends to have the lowest spot prices. Create a spot request and select the Ubuntu Server 14.04 LTS (HVM) AMI. Pick an appropriate instance type, for example c3.8xlarge. Set security group to

* SSH 22 My IP
* HTTPS 443 My IP
* Custom TCP 8888 My IP

and download a new private/public key file.

After we have `ssh` into the instance,

```sh
ssh -i [path/to/key.pem] ubuntu@[DNS]  
```

start by updating the system

```sh
sudo apt-get update
sudo apt-get dist-upgrade -y
sudo apt-get install -y git
```

and install Anaconda Python:

```sh
wget https://3230d63b5fc54e62148e-c95ac804525aac4b6dba79b00b39d1d3.ssl.cf1.rackcdn.com/Anaconda3-2.5.0-Linux-x86_64.sh
bash Anaconda3-2.5.0-Linux-x86_64.sh
source ~/.bashrc
conda update -y conda
conda update -y anaconda
conda install -y seaborn
rm Anaconda3-2.5.0-Linux-x86_64.sh
```

To install the bleeding-edge version of `mclearn` for development:

```sh
git clone https://github.com/alasdairtran/mclearn/
cd mclearn; python setup.py develop
```

Create a Jupyter notebook configuration file:

```sh
jupyter notebook --generate-config
```

Prepare a hashed password:

```sh
ipython
In [1]: from notebook.auth import passwd
In [2]: passwd()
In [3]: exit
```

Tell the notebook to communicate via a secure protocol mode by setting the `certfile` option to our self-signed certificate:

```sh
mkdir certificates
openssl req -x509 -nodes -days 365 -newkey rsa:1024 -keyout certificates/jupyter.key -out certificates/jupyter.pem
```

Open the config file

```sh
vim .jupyter/jupyter_notebook_config.py
```

and add the following lines, replacing the example hash with the one generated:

```sh
# Set options for certfile, ip, password, and toggle off browser auto-opening
c.NotebookApp.certfile = u'/home/ubuntu/certificates/jupyter.pem'
c.NotebookApp.keyfile = u'/home/ubuntu/certificates/jupyter.key'

# Set ip to '*' to bind on all interfaces (ips) for the public server
c.NotebookApp.ip = '*'
c.NotebookApp.password = u'sha1:<hash...>'
c.NotebookApp.open_browser = False

# It is a good idea to set a known, fixed port for server access
c.NotebookApp.port = 8888
```

Start the notebook with `jupyter notebook`.

Create a project folder:

```sh
sudo mkdir ~/projects
sudo chmod 777 ~/projects
```

Back in the EC2 Management Console, select the running instance. Choose Actions - Image - Create Image. This will create our own customised AMI, which we can use as a starting point for future instances (without needing to setting up everything from sratch again).

Amazon EBS provides persistent block level storage volumes for use with EC2 instances and is useful when working with large datasets. Create a new encrypted EBS volume and attach it to a running instance. Name it `/dev/xvdf`. Inside the instance, use `lsblk` to view the disk devices and their mount points. For new volumes, create an ext4 file system (this will format the volume and delete the existing data!):

```sh
sudo mkfs -t ext4 /dev/xvdf
sudo file -s /dev/xvdf
```

To mount it

```
sudo mount /dev/xvdf ~/projects
```
