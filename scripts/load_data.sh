FILENAME="AADhtxW5WM2P1b2w82rn7xyla?dl=0" 
if [ ! -f $FILENAME ]; then
	echo "ZIP file not found, will download it from dropbox"
	wget https://www.dropbox.com/sh/5p2f46s23eum86s/AADhtxW5WM2P1b2w82rn7xyla?dl=0
fi
LOGNAME="log.zip"
TARGET_REPO="../log"
/bin/mv $FILENAME* $LOGNAME
unzip $LOGNAME -d $TARGET_REPO
rm $LOGNAME 
