import 'package:flutter/material.dart';

import '../data_classes/slice_results.dart';
import '../widgets/dot_painter.dart';

class SliceViewerScreen extends StatelessWidget {
  final SliceResult result;
  const SliceViewerScreen({super.key, required this.result});

  @override
  Widget build(BuildContext context) {
    final baseBytes = result.originalPng!; //base image
    final overlayBytes = result.maskOverlayPng; //mask bytes or null if none
    final c = result.centroid; //centroid coordinates, or null if none exists

    return Scaffold(
      appBar: AppBar(
        title: Text(result.fileName),
      ),
      body: Padding(
        padding: const EdgeInsets.all(16),
        child: Column( //vertical layoutm with text and hte images
          crossAxisAlignment: CrossAxisAlignment.start, //moves all text to the left
          children: [
            Text(
              'Type: ${result.typeLabel} • ${(result.confidence * 100).toStringAsFixed(1)}%',
              style: Theme.of(context).textTheme.titleMedium,
            ),
            const SizedBox(height: 6),
            Text(
              overlayBytes == null || c == null
                  ? 'No location mask available for this slice.' //if it cant show a location mask
                  : 'Location mask confidence: ${(result.maskScore * 100).toStringAsFixed(1)}%',
            ),
            const SizedBox(height: 12),
            Expanded(
              child: Center(
                child: AspectRatio(
                  aspectRatio: 1, //makes it as close to a square as we can
                  child: Stack(
                    fit: StackFit.expand,
                    children: [
                      Image.memory(baseBytes, fit: BoxFit.contain),
                      if (overlayBytes != null) Image.memory(overlayBytes, fit: BoxFit.contain),
                      if (c != null) //if centroid exists, put it in place
                        CustomPaint(
                          painter: DotPainter(nx: c.dx, ny: c.dy),
                        ),
                    ],
                  ),
                ),
              ),
            ),
          ],
        ),
      ),
    );
  }
}