#!/bin/bash

SOURCECODE=uez56by@jeanzay:/gpfswork/rech/imi/uez56by/code/DLP/

DESTINATIONCODE=/home/tcarta/DLP

echo Transfer code ...
rsync -azvh -e "ssh -i $HOME/.ssh/id_rsa" --exclude-from=exclude_rsync_from_jeanzay.txt $SOURCECODE $DESTINATIONCODE


