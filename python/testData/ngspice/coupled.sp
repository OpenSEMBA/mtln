Coupled transmission line
R1 1 0 50
R2 2 5 50
R3 8 0 50
R4 9 0 50
R6 4 7 10
R5 3 6 25
V1 5 0 PULSE(0 1 0 1ns 1ns 499ns 1s)

p1 1 2 0 3 4 0 pline1
p2 6 7 0 8 9 0 pline2 

.control
tran 0.006ns 200ns
PLOT V(2) V(1)
WRDATA V2.txt V(2)
WRDATA V1.txt V(1)
.endc

.model pline1 cpl
+R=0 0 
+    0
+L=0.7485E-6 0.5077E-6 
+            1.0154E-6
+G=0 0 
+    0
+C=37.432E-12 -18.716E-12 
+              24.982E-12 
+length = 1.0

.model pline2 cpl
+R=0 0 
+    0
+L=0.7485E-6 0.5077E-6 
+            1.0154E-6
+G=0 0 
+    0
+C=37.432E-12 -18.716E-12 
+              24.982E-12 
+length = 1.0


.end
