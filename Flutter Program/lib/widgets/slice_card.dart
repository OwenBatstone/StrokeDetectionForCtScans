import 'package:flutter/material.dart';

class SliceCard extends StatelessWidget {
  const SliceCard({
    required this.sliceId,
    required this.imageUrl,
    this.overlay_file_url,
    super.key,
  });

  final String sliceId;
  final String imageUrl;
  final String? overlay_file_url;

  @override
  Widget build(BuildContext context) {
    return SizedBox(
      width: 110,
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.stretch,
        children: [
          // Original slice image
          ClipRRect(
            borderRadius: BorderRadius.circular(8),
            child: Image.network(
              imageUrl,
              height: 160,
              width: 110,
              fit: BoxFit.cover,
              loadingBuilder: (_, child, progress) => progress == null
                  ? child
                  : const SizedBox(
                      height: 160,
                      child: Center(
                        child: CircularProgressIndicator(strokeWidth: 2),
                      ),
                    ),
              errorBuilder: (_, __, ___) => const SizedBox(
                height: 160,
                child: ColoredBox(
                  color: Colors.black12,
                  child: Icon(Icons.broken_image, size: 32),
                ),
              ),
            ),
          ),

          const SizedBox(height: 6),

          // Overlay image
          if (overlay_file_url != null && overlay_file_url!.isNotEmpty)
            ClipRRect(
              borderRadius: BorderRadius.circular(8),
              child: Image.network(
                overlay_file_url!,
                height: 160,
                width: 110,
                fit: BoxFit.cover,
                loadingBuilder: (_, child, progress) => progress == null
                    ? child
                    : const SizedBox(
                        height: 160,
                        child: Center(
                          child: CircularProgressIndicator(strokeWidth: 2),
                        ),
                      ),
                errorBuilder: (_, __, ___) => const SizedBox(
                  height: 160,
                  child: ColoredBox(
                    color: Colors.black12,
                    child: Icon(Icons.broken_image, size: 32),
                  ),
                ),
              ),
            ),

          const SizedBox(height: 6),

          // Slice ID label
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
