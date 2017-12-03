for f in rgb/*
do
	convert $f -resize 224x224! $f

done
