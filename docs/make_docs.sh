#!/bin/bash

if [ -n "$DISTPY" ] && [ -n "$PYLINEX" ]
then
    cd $PYLINEX/docs
    pdoc --config latex_math=True --html $DISTPY/distpy $PYLINEX/pylinex --force
    cd - > /dev/null
else
    echo "DISTPY and PYLINEX environment variables must be set for the make_docs.sh script to be used to make the documentation."
fi
