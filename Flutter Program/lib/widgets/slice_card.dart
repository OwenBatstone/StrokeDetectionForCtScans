import 'package:flutter/material.dart';

class SliceCard extends StatelessWidget {
  const SliceCard({required this.sliceId, required this.imageUrl, super.key});

  final String sliceId;
  final String imageUrl;

  @override
  Widget build(BuildContext context) {
    return SizedBox(
      width: 110,
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.stretch,
        children: [
          Expanded(
            child: ClipRRect(
              borderRadius: BorderRadius.circular(8),
              child: Image.network(
                imageUrl,
                fit: BoxFit.cover,
                loadingBuilder: (_, child, progress) => progress == null
                    ? child
                    : const Center(child: CircularProgressIndicator(strokeWidth: 2)),
                errorBuilder: (_, __, ___) => const ColoredBox(
                  color: Colors.black12,
                  child: Icon(Icons.broken_image, size: 32),
                ),
              ),
            ),
          ),
          const SizedBox(height: 6),
          Text(
            sliceId,
            textAlign: TextAlign.center,
            maxLines: 2,
            overflow: TextOverflow.ellipsis,
            style: const TextStyle(fontSize: 11, fontWeight: FontWeight.w500),
          ),
        ],
      ),
    );
  }
}