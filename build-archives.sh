#!/bin/bash
jekyll build
cp -r _site/tags .
git add * -f 
git pull 
git commit -a -m "added/updated tags"
git push origin