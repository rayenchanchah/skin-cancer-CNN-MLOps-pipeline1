import subprocess
def update_dataDVC (data_ver):
 subprocess.run([ "dvc", "add", "data"])
 subprocess.run([ "git", "add", "data.dvc"])
 subprocess.run([ "git", "commit", "-m", "create new dataset version" ])
 tag = "v." + str(data_ver)
 subprocess.run([ "git", "tag", "-a", tag, "-m", "Dataset version" ])
 subprocess.run([ "git", "push"])
 subprocess.run([ "dvc", "push"])
 
if __name__ == "__main__":
 with open(param_yaml_path) as f:
    params = yaml.safe_load(f)
    data_ver = params[ 'data_version' ]
    update_dataDVC(data_ver)