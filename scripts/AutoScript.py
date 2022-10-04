import yaml
import subprocess

class AutoScript():

    def __init__(self,launch_file,config_file) -> None:
        self.launch_file = launch_file
        self.config_file = config_file

    def modify_config(self,checkpoint_path) -> None:
        with open(self.config_file,"r") as file:
            config = yaml.safe_load(file)
            config["checkpoint"] = checkpoint_path
        with open(self.config_file,"w") as file:
            yaml.dump(config)

    def launch_scitas_script(self):
        stream = subprocess.Popen(["sbatch",self.launch_file])
        

