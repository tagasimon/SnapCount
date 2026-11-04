import UIKit

enum ImageProcessing {
    /// Shrinks the photo so its longest side is at most `maxSide` pixels, bakes in the
    /// orientation, and encodes it as JPEG. Keeps uploads small without losing the detail
    /// needed to count.
    static func uploadJPEG(from image: UIImage, maxSide: CGFloat = 1600, quality: CGFloat = 0.8) -> Data? {
        let size = image.size  // already orientation-corrected, in points
        guard size.width > 0, size.height > 0 else { return nil }

        let pixelW = size.width * image.scale
        let pixelH = size.height * image.scale
        let ratio = min(1, maxSide / max(pixelW, pixelH))
        let target = CGSize(width: (pixelW * ratio).rounded(), height: (pixelH * ratio).rounded())

        let format = UIGraphicsImageRendererFormat()
        format.scale = 1
        format.opaque = true
        let resized = UIGraphicsImageRenderer(size: target, format: format).image { ctx in
            // Photos saved with a transparent background would otherwise turn black.
            UIColor.white.setFill()
            ctx.fill(CGRect(origin: .zero, size: target))
            image.draw(in: CGRect(origin: .zero, size: target))
        }
        return resized.jpegData(compressionQuality: quality)
    }
}
