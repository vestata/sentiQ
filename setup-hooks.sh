#!/bin/bash


chmod +x scripts/*

# Configure core.hooksPath to use the hooks directory
git config core.hooksPath scripts
echo "Git hooks path configured to use the 'scripts/' directory."

