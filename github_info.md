# Open your folder from Visual studio code

# Navigate to Your Project Directory: Open your terminal or command prompt and navigate to the directory where your project files are located.

# Initialize a Git Repository (If Not Already Done) by
 git  init

 # Add Your Files to the Staging Area: Use the git add command to add all files in your project directory to the staging area. You can do this with the following command:

git add filename # For all files use: git add .

# Commit Your Changes: Commit the staged files with a descriptive commit message using the git commit command:

git commit -m "message commit"


# Connect to Your GitHub Repository: If you haven't already, set up a remote connection to your GitHub repository using the git remote add command:

git remote add origin <repository_URL>


Push Your Changes to GitHub: Finally, push your committed changes to GitHub using the git push command:

git push -u origin main


#############################################################################################################
#############################################################################################################
# How to delete files

#############################################################################################################
#############################################################################################################

## Locate the File: Identify the file you want to delete from your repository.

## Delete the File Locally: If the file exists in your local repository, you can delete it using your file explorer or terminal/command prompt:

rm filename

## Replace filename.py with the name of the file you want to delete.
## Stage the Deletion: After deleting the file locally, you need to stage the deletion for commit:

## Commit the Deletion: Commit the deletion of the file

git commit -m "Delete filename "

## Commit the Deletion: Commit the deletion of the file:
git push origin main

Replace main with the name of your branch if it's different.









