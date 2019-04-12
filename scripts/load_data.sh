wget https://www.dropbox.com/sh/5p2f46s23eum86s/AADhtxW5WM2P1b2w82rn7xyla?dl=0
FILENAME="AADhtxW5WM2P1b2w82rn7xyla?dl=0" 
LOGNAME="log.zip"
TARGET_REPO="log"
mv $FILENAME $LOGNAME
unzip $LOGNAME -d $TARGET_REPO
rm $LOGNAME 
