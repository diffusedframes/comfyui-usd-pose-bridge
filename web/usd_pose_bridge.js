import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

// Scene and mapping reports are output-only DOM widgets, never prompt inputs.
app.registerExtension({
    name: "USDPoseBridge.Reports",
    setup() {
        api.addEventListener("executed", ({ detail }) => {
            const report = detail?.output?.usd_pose_report;
            if (!Array.isArray(report)) return;
            const node = app.graph?.getNodeById(detail.node);
            if (!node || !node.comfyClass?.startsWith("USDPB")) return;
            let widget = node.widgets?.find((item) => item.name === "usd_pose_report");
            if (!widget) {
                const element = document.createElement("textarea");
                element.readOnly = true;
                element.setAttribute("aria-label", "USD Pose Bridge report");
                element.style.cssText = "width:100%;height:100%;box-sizing:border-box;resize:none;" +
                    "font:12px monospace;background:var(--comfy-input-bg,#20242a);" +
                    "color:var(--input-text,#ddd);border:0;padding:8px;";
                widget = node.addDOMWidget("usd_pose_report", "text", element, { serialize: false });
                widget.computeSize = () => [300, 180];
                widget.options.serialize = false;
                node.setSize([Math.max(node.size[0], 340), Math.max(node.size[1], node.computeSize()[1])]);
            }
            widget.element.value = report.join("\n");
            node.setDirtyCanvas(true, true);
        });
    },
});
