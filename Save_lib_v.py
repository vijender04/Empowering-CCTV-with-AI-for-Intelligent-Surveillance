import subprocess
import csv
import sys

# Get the Python version
python_version = sys.version.split()[0]  # Extract only the Python version

# Get the list of installed libraries and their versions
installed_packages = subprocess.check_output(['pip', 'list']).decode('utf-8')

# Split the output into lines
lines = installed_packages.splitlines()

# Open a CSV file to write the data
with open('requirements.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    
    # Write Python version as the first row
    writer.writerow(['Python', python_version])
    
    # Write header for installed packages
    writer.writerow(['Package', 'Version'])
    
    # Skip the first two lines (which contain headers and formatting)
    for line in lines[2:]:
        package_info = line.split()
        writer.writerow(package_info)  # Write each package and version
