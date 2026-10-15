"""ComfyUI entry point for USD Pose Bridge."""

if __package__:
    from .usd_pose_bridge.nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS
else:
    # Also allow standalone test discovery from a folder with a hyphenated name.
    from usd_pose_bridge.nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

WEB_DIRECTORY = "./web"
__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]
