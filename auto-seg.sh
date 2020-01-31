#!/bin/sh
export NVM_RC_VERSION=
export VIRTUALENVWRAPPER_SCRIPT=/Library/Frameworks/Python.framework/Versions/3.5/bin/virtualenvwrapper.sh
export VIRTUALENVWRAPPER_PROJECT_FILENAME=.project
export NVM_CD_FLAGS=
export TERM=xterm-256color
export SHELL=/bin/bash
export TMPDIR=/var/folders/rd/q9_9mpf52hz9_k38mykx27q80000gn/T/
export FSLMULTIFILEQUIT=TRUE
export Apple_PubSub_Socket_Render=/private/tmp/com.apple.launchd.GZlYouGeU1/Render
export NVM_DIR=/Users/weiziyang/.nvm
export USER=weiziyang
export FSLGECUDAQ=cuda.q
export COMMAND_MODE=unix2003
export SSH_AUTH_SOCK=/private/tmp/com.apple.launchd.bVO2wnLtU9/Listeners
export __CF_USER_TEXT_ENCODING=0x1F5:0x0:0x0
export VIRTUAL_ENV=/Users/weiziyang/Documents/GitHub/code
export WORKON_HOME=/Users/weiziyang/.virtualenvs
export PATH=/Users/weiziyang/Documents/GitHub/code/bin:/usr/local/fsl/bin:/Library/Frameworks/Python.framework/Versions/3.7/bin:.:/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin:/Library/TeX/texbin:/opt/X11/bin:/Library/Frameworks/Mono.framework/Versions/Current/Commands:/Library/Frameworks/Python.framework/Versions/3.7/bin:.:/Library/Java/JavaVirtualMachines/jdk-10.0.1.jdk/Contents/Home//bin:/usr/local/mysql/bin:/Library/Java/JavaVirtualMachines/jdk-10.0.1.jdk/Contents/Home//bin:/usr/local/mysql/bin:/Library/TeX/texbin
export VIRTUALENVWRAPPER_HOOK_DIR=/Users/weiziyang/.virtualenvs
export PWD=/usr/local/fsl/bin
export FSLTCLSH=/usr/local/fsl/bin/fsltclsh
export FSLMACHINELIST=
export XPC_FLAGS=0x0
export FSLREMOTECALL=
export FSLWISH=/usr/local/fsl/bin/fslwish
export XPC_SERVICE_NAME=0
export SHLVL=1
export HOME=/Users/weiziyang
export LOGNAME=weiziyang
export FSLDIR=/usr/local/fsl
export LC_CTYPE=en_US.UTF-8
export VIRTUALENVWRAPPER_WORKON_CD=1
export FSLLOCKDIR=
export FSLOUTPUTTYPE=NIFTI_GZ
export WOKON_HOME=~/Documents/GitHub/
export SECURITYSESSIONID=186a8
export _=/usr/bin/env

from_dir='registration/'
cate_names=$(ls ${from_dir})
for each in $cate_names
do
    mkdir "segmentation/"${each}
    names=$(ls ${from_dir}/$each)
    for name in $names
    do
        full_name=${from_dir}${each}"/"${name};
        echo "processing:"${full_name};
        prefix=$(echo ${name} | cut -d '.' -f 1)
        to_name="segmentation/"${each}"/"${prefix}".nii.gz"
        if [[ ! -e "${to_name}" ]]
        then
            /usr/local/fsl/bin/fast ${full_name};
            remove_name_a="${from_dir}"${each}"/"${prefix}"_pve_0.nii.gz"
            remove_name_b="${from_dir}"${each}"/"${prefix}"_pve_1.nii.gz"
            remove_name_c="${from_dir}"${each}"/"${prefix}"_pve_2.nii.gz"
            remove_name_d="${from_dir}"${each}"/"${prefix}"_mixeltype.nii.gz"
            remove_name_e="${from_dir}"${each}"/"${prefix}"_pveseg.nii.gz"
            rm ${remove_name_a} ${remove_name_b} ${remove_name_c}  ${remove_name_d} ${remove_name_e}
            from_name="${from_dir}"${each}"/"${prefix}"_seg.nii.gz"

            mv ${from_name} ${to_name}
            echo '\n'
        else
            echo "${full_name} already exits"
        fi
    done
done
