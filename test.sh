ti1=$(date +%s)
ti2=$(date +%s)
i=$(($ti2 - $ti1 ))

while [[ "$i" -ne "3" ]]
do
	ti2=$(date +%s)
	i=$(($ti2 - $ti1 ))
done