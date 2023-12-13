import os 
import subprocess

# get list of directories
dirs = [d for d in os.listdir('.') if os.path.isdir(d)]
dirs = [d for d in dirs if not d.startswith('.')]

for d in dirs:
    print("Upgrading %s" % d)
    os.chdir(d)
    name = "prebuilt-"+d.replace('_', '-')
    subprocess.call(['cerebrium', 'upgrade-yaml', '--name', name, ])
    if os.path.exists('config.yaml.legacy'):
        os.remove('config.yaml.legacy')
    os.chdir('..')
