import 'package:flutter/material.dart';

import '../data_classes/slice_results.dart';
import '../widgets/dot_painter.dart';
class SliceViewerScreen extends StatefulWidget {
  final SliceResult result;
  const SliceViewerScreen({super.key, required this.result});

  @override
  State<SliceViewerScreen> createState() => _SliceViewerScreenState();
  }

class _SliceViewerScreenState extends State<SliceViewerScreen>{
  bool _showOverlay = true;
  @override
  Widget build(BuildContext context) {
    final result= widget.result;
    final baseBytes = result.originalPng!; //base image
    final overlayBytes = result.maskOverlayPng; //mask bytes or null if none

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
              overlayBytes == null
                  ? 'No location mask available for this slice.'
                  : 'Location mask: ${(result.maskScore * 100).toStringAsFixed(1)}%',
            ),
            const SizedBox(height: 12),

            if (overlayBytes != null)
              Wrap(
                spacing: 8,
                runSpacing: 8,
                children: [
                  FilterChip(
                  selected: _showOverlay,
                  label: const Text('Show Overlay'),
                  onSelected: (value) {
                    setState(() {
                      _showOverlay = value;
                    });
                  },
                ),
            ],
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
                      if (_showOverlay && overlayBytes != null)
                        Image.memory(overlayBytes, fit: BoxFit.contain),
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