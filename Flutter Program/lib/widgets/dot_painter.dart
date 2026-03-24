import 'package:flutter/material.dart';
import 'dart:math' as math;

class DotPainter extends CustomPainter {
  //for making the dot based off the normalized coordinates
  //normalized coordinates
  final double nx;
  final double ny;

  //painter constructor
  DotPainter({required this.nx, required this.ny});

  @override
  void paint(Canvas canvas, Size size) {
    //converts the normalized coords into the pixel coords
    final p = Offset(nx.clamp(0, 1) * size.width, ny.clamp(0, 1) * size.height);

    final paintOuter = Paint()
      ..color = Colors.white.withValues(alpha: .95); //white ouutline of dot
    final paintInner = Paint()
      ..color = Colors.redAccent.withValues(alpha: .95); //red inside of dot

    canvas.drawCircle(p, math.max(6, size.shortestSide * 0.02), paintOuter);
    canvas.drawCircle(p, math.max(3.5, size.shortestSide * 0.012), paintInner);
  }

  @override
  bool shouldRepaint(covariant DotPainter oldDelegate) {
    //if dot moves repaint
    return oldDelegate.nx != nx || oldDelegate.ny != ny;
  }
}
