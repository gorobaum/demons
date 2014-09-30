rgb(r,g,b) = int(r)*65536 + int(g)*256 + int(b)
set size square
plot filename using 1:2:3:4:(rgb($5,$6,$7)) with vectors lw 3 lc rgb variabl
pause -1
