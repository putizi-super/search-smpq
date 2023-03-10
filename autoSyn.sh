# from https://blog.csdn.net/jasonLee_lijiaqi/article/details/79947741

# the method for “warning: LF will be replaced by CRLF”

#提交时转换为LF，检出时转换为CRLF from https://blog.csdn.net/u012757419/article/details/105614028
# git config --global core.autocrlf true
git status  
git add *  
git commit -m 'add some code from Windows'
# git commit -m 'add some results from Server'
# git pull --rebase origin master   #domnload data
git pull --rebase origin vit   #domnload data
git push origin vit            #upload data
git stash pop